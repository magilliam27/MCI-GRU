# Blind reviewer instructions for 8-K events, version 1

You are one of three independent reviewers labelling events that companies
reported in 8-K filings. Your answers become part of an answer key, so they
must be your own reading of each item and nothing else.

## Rules

- Read exactly one file: `$items_path`. Do not open, list or search any other
  file or folder, and do not use the web. Other files may hold other people's
  answers, and seeing them would spoil the key.
- Judge each item from a shareholder's point of view, using only its context
  and text. Do not use what you know about the company, or anything that
  happened after the filing, such as how the stock moved or how the matter
  turned out.
- Answer every item. When you are unsure, give your best reading and say why in
  the note; never leave a question blank.
- Write one file: `$answers_path`. Write nothing else.

## The task: $title

$intro

Answer these questions for every item. For each one, write exactly one of its
options, as shown.

$questions

## Answer file

A CSV with this header row, then one row per item, in any order:

```
$header
```

- `item_id` is the id shown on the item, such as `$example_id`.
- `note` is one short sentence on what decided your answers. Quote it if it
  contains a comma.

There are $item_count items. Check that the file has $item_count rows before
you finish, then reply with one line: the number of rows written.
