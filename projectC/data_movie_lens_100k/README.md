# Contents

## movie_info.csv

Contains information about each movie (one per row of CSV file).

Column description:

```
item_id : int
  Unique id for each movie in our released version of dataset for project C
  Will be an integer between 0 ... 1681 (inclusive)
title : str
  Name of the movie
release_year : int
  Release year of the movie (e.g. 1998 or 2004)
orig_item_id : int
  Original id for this movie in the original movie_lens_100k database.
  Not needed at all for project C
```


## user_info.csv

Contains information about each user (one per row of CSV file).

Column description:

```
user_id : int
  Unique id for each user in our released version of dataset for project C
  Will be an integer between 0 ... 942 (inclusive)
age : int
  Age of the user in years (whole number)
is_male : int
  Binary indicator of if the user was marked as male (value 1) or not (value 0).
  From the original metadata from movie lens project.
orig_user_id : int
  Original id for this user in the original movie_lens_100k database.
  Not needed at all for project C, could be used if this dataset needed to be relinked to original
```
