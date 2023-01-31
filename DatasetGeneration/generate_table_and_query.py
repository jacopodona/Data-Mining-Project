import csv
import random
import json
import os
import argparse
import pandas as pd
# import generate_user_list
# import generate_utility_matrix

ASSET_DIR = 'assets'
TABLE_OUTPUT_DIR = 'tables'
QUERY_OUTPUT_DIR = 'queries'
MUSIC_ITEMS_LEN = 10000
MOVIE_ITEMS_LEN = 5000
PEOPLE_ITEMS_LEN = 50000


def generateMusicTable():
    header = ['id', 'name', 'artist', 'genre', 'year']  # Add nationality to songs?

    # Load nouns for song name
    with open(os.path.join(ASSET_DIR, 'nouns.json')) as f:
        data = json.load(f)
        nouns = []
        for i in data:
            nouns.append(i['noun'])
    print('Nouns:', len(nouns))

    # Load genres
    with open(os.path.join(ASSET_DIR, 'music_genres.json')) as f:
        genres = json.load(f)
    print('Genres:', len(genres))

    # Load first names for artist name
    with open(os.path.join(ASSET_DIR, 'first_names.json')) as f:
        data = json.load(f)
        first_names = []
        for i in data:
            first_names.append(i['first_name'])
    print('Names:', len(first_names))

    year = ['60s', '70s', '80s', '90s', '2000s', '2010s', '2020s']

    id = 0
    songs = []
    for i in range(0, MUSIC_ITEMS_LEN):
        item_id = id
        item_name = random.choice(nouns)
        item_genre = random.choice(genres)
        item_duration = random.choice(year)
        item_artist = random.choice(first_names)
        # print(item_id,'\n',item_name,'\n',item_genre,'\n',item_duration,'\n',item_nationality,'\n','='*35)
        id += 1
        item = [item_id, item_name, item_artist, item_genre, item_duration]
        songs.append(item)

    # open the file in the write mode
    with open(os.path.join(TABLE_OUTPUT_DIR, 'music.csv'), 'w',newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(header)

        writer.writerows(songs)


def generateMoviesTable():
    header = ['id', 'name', 'genre', 'duration', 'nationality']

    # Load genres
    with open(os.path.join(ASSET_DIR, 'movie_genres.json')) as f:
        data = json.load(f)
        genres = []
        for i in data:
            genres.append(i['movie_genre'])
    print('Genres:', len(genres))

    # Load nations
    with open(os.path.join(ASSET_DIR, 'nations.json')) as f:
        data = json.load(f)
        states = []
        for i in data:
            states.append(i['state'])
    print('States:', len(states))

    # Load nouns
    with open(os.path.join(ASSET_DIR, 'nouns.json')) as f:
        data = json.load(f)
        nouns = []
        for i in data:
            nouns.append(i['noun'])
    print('Nouns:', len(nouns))

    id = 0
    movies = []
    for i in range(0, MOVIE_ITEMS_LEN):
        item_id = id
        item_name = random.choice(nouns) + ' ' + random.choice(nouns)
        item_genre = random.choice(genres)
        item_duration = random.randint(90, 240)
        item_nationality = random.choice(states)
        # print(item_id,'\n',item_name,'\n',item_genre,'\n',item_duration,'\n',item_nationality,'\n','='*35)
        id += 1
        item = [item_id, item_name, item_genre, item_duration, item_nationality]
        movies.append(item)

    # open the file in the write mode
    with open(os.path.join(TABLE_OUTPUT_DIR, 'movies.csv'), 'w',newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(header)

        writer.writerows(movies)


def generatePeopleTable():
    header = ['id', 'first_name', 'last_name', 'age', 'job', 'nationality']

    # Load first names
    with open(os.path.join(ASSET_DIR, 'first_names.json')) as f:
        data = json.load(f)
        first_names = []
        for i in data:
            first_names.append(i['first_name'])
    print('First names:', len(first_names))

    # Load last names
    with open(os.path.join(ASSET_DIR, 'last_names.json')) as f:
        data = json.load(f)
        last_names = []
        for i in data:
            last_names.append(i['last_name'])
    print('Last names:', len(last_names))

    # Load nations
    with open(os.path.join(ASSET_DIR, 'nations.json')) as f:
        data = json.load(f)
        states = []
        for i in data:
            states.append(i['state'])
    print('States:', len(states))

    # Load job titles
    with open(os.path.join(ASSET_DIR, 'job_title.json')) as f:
        job_titles = json.load(f)
    print('Job Titles:', len(job_titles))

    # Load job area
    with open(os.path.join(ASSET_DIR, 'job_area.json')) as f:
        job_areas = json.load(f)
    print('Job Areas:', len(job_areas))

    id = 0
    people = []
    for i in range(0, PEOPLE_ITEMS_LEN):
        item_id = id
        item_first_name = random.choice(first_names)
        item_last_name = random.choice(last_names)
        item_age = random.randint(18, 80)
        item_nationality = random.choice(states)
        item_job = random.choice(job_areas) + ' ' + random.choice(job_titles)
        # print(item_id,'\n',item_name,'\n',item_genre,'\n',item_duration,'\n',item_nationality,'\n','='*35)
        id += 1
        item = [item_id, item_first_name, item_last_name, item_age, item_job, item_nationality]
        people.append(item)

    # open the file in the write mode
    with open(os.path.join(TABLE_OUTPUT_DIR, 'people.csv'), 'w',newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(header)

        writer.writerows(people)


def generateQueries(table, numquery):
    '''
    Takes as input a csv table path and generates random queries iterating through the table items

    :param table: 'movies.csv' or 'people.csv' or 'music.csv'
    :param numquery: Number of queries to generate
    :return:
    '''
    tablename = table.split('.')[0]  # get table name from table.csv

    if tablename == 'movies':
        header = ['id', 'name', 'genre', 'duration', 'nationality']
        min = 1
        max = 4
    elif tablename == 'people':
        header = ['id', 'first_name', 'last_name', 'age', 'job', 'nationality']
        min = 1
        max = 5
    elif tablename == 'music':
        header = ['id', 'name', 'artist', 'genre', 'year']
        min = 1
        max = 4

    with open(os.path.join(TABLE_OUTPUT_DIR, table)) as f:
        df = pd.read_csv(f)

    queries = set()
    while len(queries) != numquery:
        query = ''
        randomRow = df.iloc[random.randint(0, len(df))]
        i = random.randint(1, 4)  # Generate random length query
        q_id = 'Q' + str(len(queries))
        if i == 1:  # Generate random query of length 1
            h = random.randint(min, max)
            query = q_id + ', ' + header[h] + '=' + str(randomRow[h])
            pass
        elif i == 2:  # Generate random query of length 2
            h1 = 1
            h2 = 1
            while h1 == h2:
                h1 = random.randint(min, max)
                h2 = random.randint(min, max)
            query = q_id + ', ' + header[h1] + '=' + str(randomRow[h1]) + ', ' + header[h2] + '=' + str(randomRow[h2])
            pass
        elif i == 3:  # Generate random query of length 3
            h1 = 1
            h2 = 1
            h3 = 1
            while h1 == h2 or h2 == h3 or h1 == h3:
                h1 = random.randint(min, max)
                h2 = random.randint(min, max)
                h3 = random.randint(min, max)
            query = q_id + ', ' + header[h1] + '=' + str(randomRow[h1]) + ', ' + header[h2] + '=' + str(
                randomRow[h2]) + ', ' + header[h3] + '=' + str(randomRow[h3])
            pass
        elif i == 4:  # Generate random query of length 4
            h1 = 1
            h2 = 1
            h3 = 1
            h4 = 1
            while h1 == h2 or h2 == h3 or h1 == h3 or h1 == h4 or h2 == h4 or h3 == h4:
                h1 = random.randint(min, max)
                h2 = random.randint(min, max)
                h3 = random.randint(min, max)
                h4 = random.randint(min, max)
            query = q_id + ', ' + header[h1] + '=' + str(randomRow[h1]) + ', ' + header[h2] + '=' + str(
                randomRow[h2]) + ', ' + header[h3] + '=' + str(randomRow[h3]) + ', ' + header[h4] + '=' + str(
                randomRow[h4])
            pass
        queries.add(str(query))

    # open the file in the write mode
    with open(os.path.join(QUERY_OUTPUT_DIR, tablename + '.csv'), 'w',newline='') as f:
        # create the csv writer
        writer = csv.writer(f)

        for row in queries:
            writer.writerow([row])


def generateTable(domain):
    '''
    :param domain: Parameter for the table domain, available domains are 'music', 'movies', 'people'.
                    Each domain has different properties
    :return: Generates <domain>.csv file
    '''

    if domain == 'music':
        generateMusicTable()
    elif domain == 'movies':
        generateMoviesTable()
    elif domain == 'people':
        generatePeopleTable()


def main():
    parser = argparse.ArgumentParser(description="Script for table and query generation")
    parser.add_argument("--domain", type=str, default='people',
                        help="Domain for the dataset (Available domains: movies, music, people)")
    args = parser.parse_args()

    domain = args.domain

    generateTable(domain)
    tableFile = domain + '.csv'
    generateQueries(tableFile, 500)


if __name__ == '__main__':
    main()
