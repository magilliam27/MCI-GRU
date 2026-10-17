# Blind reviewer instructions, version 2

You are one of three independent reviewers labelling pairs of risk factors from
companies' 10-K filings. Your answers become part of an answer key, so they
must be your own reading of the two texts and nothing else.

## Rules

- Read exactly one file: `$items_path`. Do not open, list or search any other
  file or folder, and do not use the web. Other files may hold other people's
  answers, and seeing them would spoil the key.
- Judge each item on its own two texts. Do not use what you know about the
  company or later events.
- Answer every item. When you are unsure, give your best reading and say why in
  the note; never leave an applicable question blank.
- Write one file: `$answers_path`. Write nothing else.

## The task: $title

$intro

Answer these questions for every item. Each answer is `yes` or `no`, lower
case, unless the question lists its own options; then write exactly one of
those options, as shown.

$questions

## Answer file

A CSV with this header row, then one row per item, in any order:

```
$header
```

- `item_id` is the id shown on the item, such as `$example_id`.
- Leave a question's cell empty only when its condition above says to.
- `note` is one short sentence on what decided your answers. Quote it if it
  contains a comma.

There are $item_count items. Check that the file has $item_count rows before
you finish, then reply with one line: the number of rows written.
