The repository contains the code for the dataset generation and the algorithm used for the project.

The algorithm is divided into several scripts:

1) generate_table_and_query generates tables containing items and queries returing each at least one result.
2) generate_user_list generates a list of users with unique id.
3) generate_utility_matrix generates a utility matrix given the table, queries and user list as input.
4) baseline contains our baseline algorithm.
5) final_solution.py contains our final proposed method for solving the utility matrix filling problem.
6) utils and input_utils contain methods for dealing with input files and evaluation steps.
7) get_similarity_scores contains methods for performing Jaccard Similarity using polars dataframe.
8) svd_NOT_WORKING contains our failed trial in using a Matrix Factorization Algorithm for dimensionality reduction, it does not run.

The project is implemented in python, in requirements.txt we share the pip freeze requirements of our conda environment

----------------------------------------------------------- DATASET GENERATION ------------------------------------------------

For coherent results, it's mandatory to run the 3 scripts in the previously described order

For the project, we decided to generate 3 datasets, each representing a particular scenario where query reccomendation system could be integrated.

- People represents a Linkedin-like social network, where we have a large amount of items in the table and a lesser number of users executing queries.
- Music represents a Spotify-like media platform, where the number of items and the number of the user is balanced.
- Movies represents a Netflix-like platform, where we have a restricted number of items but a large number of users.

To generate the utility matrix, users have a small subset of query rated using a uniform probability.
From there, another subset of rated query for each user is generated comparing the new query to rate and the previously rated ones, using a normal distribution with variance being inversely proportional to the degree of similarity of the 2 queries (similar queries --> small variance in vote distribution)

To generate the dataset for each domain, run:

1) python generate_table_and_query.py --domain people #(or music or movies)
2) python generate_user_list.py
3) generate_utility_matrix.py --domain people #(or music or movies)(extra parameters are required only if query.csv or user files are renamed)

Before running, it's necessary to have 4 folders in the same directory of the scripts called 'queries','tables', 'users' and 'utility_matrices'.

For generating the items, we used various json files containing names, movie/song genres, nationalities etc...
We placed them inside the assets folder and are necessary to generate the tables.

----------------------------------------------------------- ALGORITHM ---------------------------------------------------------------------

baseline.py runs our baseline onto the chosen scenario (to modify the scenario, replace the matrix variable containing the utility matrix path inside the main method).

final_solution.py runs our proposed solution onto the chosen scenario (to modify the scenario, replace the domain variable inside the main method).

It is possible to run our algorithm with personalized data, however it will need to be placed inside the appropriate folder (data/queries,data/users,...). For easier testing phase, we reccomend following the naming style we used.

We automatically perform a validation step inside the algorithm, masking 20% of the queries for 30% of the users and perform the metrics described on the report on the missing data

For further questions:
jacopo.dona@studenti.unitn.it
jacopo.clocchiatti@studenti.unitn.it