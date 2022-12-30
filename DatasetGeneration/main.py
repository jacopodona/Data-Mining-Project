import csv
import random
import json
import argparse


def generateMusicTable():
    header = ['id', 'name', 'artist', 'genre', 'duration']


def generateMoviesTable():
    header = ['id', 'name', 'genre', 'duration', 'nationality']


def generatePeopleTable():
    header = ['id', 'name', 'age', 'job', 'nationality']


def generateTable(domain):
    '''
    :param domain: Parameter for the table domain, available domains are 'music', 'movies', 'people'. Each domain has different properties
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
    parser.add_argument("--domain", type=str, default='music',
                        help="Domain for the dataset (Available domains: movies, music, people)")
    parser.add_argument("--table", type=bool, default=False, help="Flag for generating tables")
    parser.add_argument("--utility", type=bool, default=False, help="Flag for generating utility matrix")
    args = parser.parse_args()

    domain = args.domain
    table = args.table
    utility = args.utility

    print(domain, table, utility)

    # generateTable('music')


if __name__ == '__main__':
    main()
