import csv
import random
import json


def generateMusicTable():
    header=['id','name','artist','genre','duration']


def generateMoviesTable():
    header=['id','name','genre','duration','nationality']


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
    generateTable('music')



if __name__ == '__main__':
    main()
    print('Prova')

