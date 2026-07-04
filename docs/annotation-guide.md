# Annotation guide

## What to box

Choose the class before drawing each box:

- `painted_number`: draw one tight box around the complete painted identifier;
- `license_plate`: draw one tight box around the complete physical plate.

Include only a small amount of surrounding background.

Include:

- all digits belonging to the identifier;
- worn or dirty identifiers that a person can still interpret;
- partially occluded examples when the visible evidence is useful.

Exclude:

- phone numbers;
- safety instructions;
- graffiti;
- logos and company names;
- identifiers that are completely unreadable.

## Transcription

- Enter only the identifier characters, without spaces or punctuation.
- For `painted_number`, use digits `0-9`.
- For `license_plate`, use uppercase letters and digits without spaces or punctuation.
- Do not guess. Mark uncertain text as unreadable and leave transcription
  empty.
- Preserve leading zeroes.

## Review states

- `draft`: created but not independently checked.
- `verified`: box and transcription checked.
- `rejected`: annotation should not enter a dataset release.

## Video frames

Do not annotate hundreds of nearly identical adjacent frames. Select diverse
frames covering changes in position, scale, angle, blur, occlusion, and light.
Keep every frame from one visit in the same dataset split.
