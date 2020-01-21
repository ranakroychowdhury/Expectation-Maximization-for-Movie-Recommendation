# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:52:18 2019

@author: Ranak Roy Chowdhury
"""
import math
import numpy as np
            
            
def readFiles():
    with open("hw8_ids.txt", "r") as file:
        ids = [line.split()[0] for line in file]
    with open("hw8_movies.txt", "r") as file:
        movies = [line.split()[0] for line in file]
    with open("hw8_ratings.txt", "r") as file:
        ratings = [[x for x in line.split()] for line in file]
    ratings = np.array(ratings)
    with open("hw8_probZ_init.txt", "r") as file:
        probZ = [float(line.split()[0]) for line in file]
    with open("hw8_probR_init.txt", "r") as file:
        probR = [[float(x) for x in line.split()] for line in file]
    probR = np.array(probR)
    return ids, movies, ratings, probZ, probR
    

def sanityCheck(movies, ratings):
    saw = (ratings != '?').sum(0)
    recommend = (ratings == '1').sum(0)
    mean_popularity = np.true_divide(recommend, saw)
    print('Most popular movie is ' + movies[np.argmax(mean_popularity)])
    print('Least popular movie is ' + movies[np.argmin(mean_popularity)])
    

def computeRho(ratings, probZ, probR):
    users, movies = ratings.shape
    prob_R = probR.transpose()
    rho = []
    for i in range(len(probZ)):
        l = []
        for j in range(users):
            prod = 1
            for k in range(movies):
                if ratings[j][k] == '1':
                    prod *= prob_R[i][k]
                elif ratings[j][k] == '0':
                    prod *= 1 - prob_R[i][k]
            l.append(prod)
        rho.append(l)
    for i in range(len(probZ)):
        for j in range(users):
            rho[i][j] = rho[i][j] * probZ[i]
    return rho


def computeLogLikelihood(rho):
    r = np.array(rho)
    s = r.sum(0)
    L = 0
    for i in range(len(s)):
        L += math.log(s[i])
    return L/len(s)
    
    
def expectation(rho):
    rho_it = np.array(rho)
    genres, users = rho_it.shape
    s = rho_it.sum(0)
    for i in range(users):
        for j in range(genres):
            rho_it[j][i] = rho[j][i] / s[i]
    return rho_it


def computeProbZ(probZ, rho_it):
    genres, users = rho_it.shape
    denom = list(rho_it.sum(1))
    for i in range(genres):
        probZ[i] = denom[i] / users
    return probZ, denom
    

def computeProbR(ratings, probR, rho_it, denom):
    genres, users = rho_it.shape
    users, movies = ratings.shape
    movies, genres = probR.shape
    for i in range(genres):
        for j in range(movies):
            s = 0
            for k in range(users):
                if ratings[k][j] == '1':
                    s += rho_it[i][k]
                elif ratings[k][j] == '?':
                    s += rho_it[i][k] * probR[j][i]
            probR[j][i] = s / denom[i]
    return probR

    
def maximization(ratings, probZ, probR, rho_it):
    probZ, denom = computeProbZ(probZ, rho_it)
    probR = computeProbR(ratings, probR, rho_it, denom)
    return probZ, probR
    
    
def EM(ratings, probZ, probR, iteration):
    log_likelihood = []
    for i in range(iteration):
        rho = computeRho(ratings, probZ, probR)
        L = computeLogLikelihood(rho)
        log_likelihood.append(L)
        rho_it = expectation(rho)
        probZ, probR = maximization(ratings, probZ, probR, rho_it)
    return log_likelihood, probZ, probR, rho_it

    
def displayLogLikelihood(log_likelihood, l):
    for i in range(len(l)):
        print(log_likelihood[l[i]])
    

def recommend(ids, movies_names, ratings, probZ, probR, rho_it):
    users, movies = ratings.shape
    genres, users = rho_it.shape
    idx = ids.index('A53317421')
    l = []
    for i in range(movies):
        if ratings[idx][i] == '?':
            l.append(i)
    rate = []
    for i in range(len(l)):
        r = 0
        for k in range(genres):
            r += rho_it[k][idx] * probR[l[i]][k]
        rate.append(r)
    dic = {}
    for i in range(len(l)):
        dic[movies_names[l[i]]] = rate[i]
    sorted_dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    print(sorted_dic)
    
    
if __name__ == "__main__":
    print("Reading Files")
    ids, movies, ratings, probZ, probR = readFiles()
    sanityCheck(movies, ratings)
    iteration = 257
    log_likelihood, probZ, probR, rho_it = EM(ratings, probZ, probR, iteration)
    l = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    displayLogLikelihood(log_likelihood, l)
    recommend(ids, movies, ratings, probZ, probR, rho_it)