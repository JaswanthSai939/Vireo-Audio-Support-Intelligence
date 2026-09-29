# Repeat Detector Validation

## Sample

Review the 50 rows in:

`repeat_validation_sample.csv`

Each row contains:

- Current customer message
- Previous customer message
- Time gap
- Text similarity
- Product
- Category
- Previous product
- Previous category

## Label each row

Use exactly one of:

### TRUE

Use `TRUE` when the current contact appears to concern
the same underlying customer issue as the previous contact.

### FALSE

Use `FALSE` when the customer is contacting support about
a different issue, even if the customer/product/category is
the same.

### UNCLEAR

Use `UNCLEAR` when the available information is insufficient
to determine whether the issue is actually the same.

## Important

Do not label a ticket TRUE merely because:

- the customer is the same;
- the product is the same;
- the category is the same;
- the text sounds similar.

The underlying problem should be substantially the same.

## After reviewing

Save the CSV with your labels.

Then run:

`python src/04_validate_repeat_detection.py`

again.

The script will calculate:

- total reviewed;
- TRUE;
- FALSE;
- UNCLEAR;
- precision;
- error rate.
