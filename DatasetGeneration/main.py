import csv
import random
import json
import os
import argparse

ASSET_DIR='assets'
OUTPUT_DIR='outputs'
MOVIE_ITEMS_LEN=3000


def generateMusicTable():
    header=['id','name','artist','genre','duration']


def generateMoviesTable():
    header=['id','name','genre','duration','nationality']

    #Load genres
    with open(os.path.join(ASSET_DIR,'movie_genres.json')) as f:
        data = json.load(f)
        genres=[]
        for i in data:
            genres.append(i['movie_genre'])
    print('Genres:',len(genres))

    # Load nations
    with open(os.path.join(ASSET_DIR, 'nations.json')) as f:
        data = json.load(f)
        states = []
        for i in data:
            states.append(i['state'])
    print('States:',len(states))

    # Load nouns
    with open(os.path.join(ASSET_DIR, 'nouns.json')) as f:
        data = json.load(f)
        nouns = []
        for i in data:
            nouns.append(i['noun'])
    print('Nouns:',len(nouns))

    id=0
    movies=[]
    for i in range(0,MOVIE_ITEMS_LEN):
        item_id=id
        item_name=random.choice(nouns)+' '+random.choice(nouns)
        item_genre=random.choice(genres)
        item_duration=random.randint(90,240)
        item_nationality=random.choice(states)
        #print(item_id,'\n',item_name,'\n',item_genre,'\n',item_duration,'\n',item_nationality,'\n','='*35)
        id += 1
        item=[item_id,item_name,item_genre,item_duration,item_nationality]
        movies.append(item)

    # open the file in the write mode
    with open(os.path.join(OUTPUT_DIR,'movies.csv'), 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(header)

        writer.writerows(movies)


def generatePeopleTable():
    header = ['id', 'name', 'age', 'job', 'nationality']


def generateTable(domain):
    '''
    :param domain: Parameter for the table domain, available domains are 'music', 'movies', 'people'. Each domain has different properties
    :return: Generates <domain>.csv file
    '''

    if domain=='music':
        generateMusicTable()
    elif domain=='movies':
        generateMoviesTable()
    elif domain=='people':
        generatePeopleTable()

def main():
    parser = argparse.ArgumentParser(description="Script for table and query generation")
    parser.add_argument("--domain", type=str, default='',help="Domain for the dataset (Available domains: movies, music, people)")
    parser.add_argument("--table", type=bool, default=False, help="Flag for generating tables")
    parser.add_argument("--utility", type=bool, default=False, help="Flag for generating utility matrix")
    args = parser.parse_args()

    domain = args.domain
    table = args.table
    utility = args.utility

    #print(domain,table,utility)

    generateTable(domain)



if __name__ == '__main__':
    main()

