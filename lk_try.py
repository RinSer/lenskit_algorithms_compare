import pandas as pd
import matplotlib.pyplot as plt
from lenskit.datasets import ML1M, MovieLens
from lenskit.algorithms import Recommender, basic, item_knn, user_knn, als, funksvd
from lenskit.crossfold import partition_users, SampleFrac
from lenskit import batch, topn, util

def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

def eval_algos(ratings, algorithms):
    all_recs = []
    test_data = []
    for train, test in partition_users(
        ratings[['user', 'item', 'rating']], 5, SampleFrac(0.2)):
        test_data.append(test)
        for key, value in algorithms.items():
            all_recs.append(eval(key, value, train, test))
    all_recs = pd.concat(all_recs, ignore_index=True)
    print('Algorithms\' results table head:')
    print(all_recs.head())
    test_data = pd.concat(test_data, ignore_index=True)
    return all_recs, test_data

def eval_ndcg(all_recs, test_data):
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(all_recs, test_data)
    print('Normalized Discounted Cumulative Gains table head:')
    print(results.head())
    return results.groupby('Algorithm').ndcg.mean()

def plot_comparison(algorithm_means):
    plt.bar(x=algorithm_means.index.values, height=list(algorithm_means))
    plt.show()

def test_alogrithms():
    data = MovieLens('ml-latest-small')
    #data = ML1M('ml-1m')
    ratings = data.ratings
    print('Initial ratings table head:')
    print(ratings.head())
    algorithms = {
        'Bias' : basic.Bias(damping=5),
        'Popular' : basic.Popular(),
        'ItemItem' : item_knn.ItemItem(20),
        'UserUser' : user_knn.UserUser(20),
        'BiasedMF' : als.BiasedMF(50),
        'ImplicitMF' : als.ImplicitMF(50),
        'FunkSVD' : funksvd.FunkSVD(50)
    }
    all_recs, test_data = eval_algos(ratings, algorithms)
    ndcg_means = eval_ndcg(all_recs, test_data)
    print('NDCG means:')
    print(ndcg_means)
    plot_comparison(ndcg_means)

if __name__ == '__main__':
    test_alogrithms()