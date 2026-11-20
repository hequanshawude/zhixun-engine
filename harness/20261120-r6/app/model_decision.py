"""Request a bounded, validated Qwen decision through the host gateway."""
import sys


def brief_reason(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Model gave no reasoning')
    reason = ' '.join(value.split())
    try:
        encoded = reason.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ValueError('Model reasoning is not valid Unicode') from exc
    # The SDK limits each record field to 4,000 bytes. Leave room for the gate and receipt.
    if len(encoded) <= 3000:
        return reason
    return encoded[:2997].decode('utf-8', errors='ignore').rstrip() + '...'


def _finish_reason(receipt):
    choices = receipt.response.get('choices')
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return str(choices[0].get('finish_reason') or 'unknown')[:80]
    return 'unknown'


def request_decision(rnd, messages, parse):
    for attempt in range(2):
        # A transport failure may leave the host outcome unknown. Only a received reply is reparsed.
        receipt = rnd.model(messages, max_tokens=rnd.max_output_tokens,
                            temperature=0, enable_thinking=False)
        try:
            return parse(receipt.text), receipt
        except ValueError as exc:
            detail = (f'Qwen decision unusable (attempt={attempt + 1}, '
                      f'finish_reason={_finish_reason(receipt)}, '
                      f'max_tokens={rnd.max_output_tokens}): {exc}')
            if attempt == 1:
                raise ValueError(detail) from exc
            print(detail, file=sys.stderr, flush=True)
            messages = [*messages, {'role': 'user', 'content':
                'The last reply did not yield a valid action. Return one short JSON object only. '
                'Use the required keys and one allowed value for each option. Keep the reason to one sentence. '
                'No Markdown or other text.'}]
