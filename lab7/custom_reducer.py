import math
import numpy as np


class Reducer:

    def __init__(self, reqressor_type, error_metric):
        self._regressor_type = reqressor_type
        self._error_metric = error_metric

    def reduce(self, x, y, max_error=1e-2):
        x = np.array(x)
        data = x.copy()
        removed_indexes = []
        for _ in range(len(data[0])):
            for i in range(len(data[0])):
                if i in removed_indexes:
                    continue
                if len(removed_indexes) + 1 >= len(data[0]):
                    break
                regressor = self._regressor_type()
                learn = self._create_array(data, removed_indexes, i)
                regressor.fit(learn, y)
                predicted = regressor.predict(learn)
                error = self._error_metric(y, predicted)
                if math.fabs(error) < max_error:
                    removed_indexes.append(i)
        return removed_indexes

    def _create_array(self, data, reduced_indexes: list, reduce_index):
        indexes = reduced_indexes[:]
        indexes.append(reduce_index)
        return np.delete(data, indexes, axis=1)
