# -*- coding: utf-8 -*-
import numpy as np
import pickle
import time


class LshModel:
    def __init__(self, nbTables, nbBitsPerTable, nbDimensions, model_path=None, normals_path=None, thresholds_path=None):
        self.nbTables = nbTables
        self.nbBitsPerTable = nbBitsPerTable
        self.nbDimensions = nbDimensions
        self.normals = []
        self.thresholds = []
        assert (normals_path is not None and thresholds_path is not None) or model_path is not None, 'no model or record'
        if normals_path is not None and thresholds_path is not None:
            self.load_lsh_model(normals_path, thresholds_path)
        else:
            self.make_lsh_model(model_path)
    
    def make_lsh_model(self, model_path):
        start_time = time.time()
        modelData = pickle.load(open(model_path, 'rb'))
        print('load %s using: %.3fs' % (model_path, time.time()-start_time))
        self.midpoints = []
        self.normals = []
        self.thresholds = []
        totalSize = 2*self.nbTables*self.nbBitsPerTable
        interval = 2 * self.nbBitsPerTable
        for i in range(0, totalSize, interval):
            vector_sample_q = modelData[i:i+interval:2]
            vector_sample_a = modelData[i+1:i+1+interval:2]
            cur_midpoints = (vector_sample_q + vector_sample_a) / 2
            cur_normals = vector_sample_q - cur_midpoints
            cur_thresholds = np.zeros(self.nbBitsPerTable)
            for j in range(self.nbBitsPerTable):
                cur_thresholds[j] = cur_normals[j].dot(cur_midpoints[j])
            
            self.normals.append(cur_normals)
            self.thresholds.append(cur_thresholds)

    def get_lsh_hashes(self, vector):
        hashes=dict()
        for normal, thresholds in zip(self.normals, self.thresholds):
            hsh = 0
            dot = vector.dot(normal.T)  # shape (nb_bits,)
            for i, (d, t) in enumerate(zip(dot, thresholds)):
                if d > t:
                    hsh += 2 ** i
            hashes[len(hashes)] = hsh
        return hashes

    def save_lsh_model(self, normals_path, thresholds_path):
        pickle.dump(self.normals, open(normals_path, 'wb'))
        pickle.dump(self.thresholds, open(thresholds_path, 'wb'))

    def load_lsh_model(self, normals_path, thresholds_path):
        self.normals = pickle.load(open(normals_path, 'rb'))
        assert np.array(self.normals).shape == (self.nbTables, self.nbBitsPerTable, self.nbDimensions)
        self.thresholds = pickle.load(open(thresholds_path, 'rb'))
        assert np.array(self.thresholds).shape == (self.nbTables, self.nbBitsPerTable)