import numpy as np
import  collections

class DataCenter:
    def __init__(self, config):
        self.config = config

    def load_dataSet(self, dataSet='cora'):
        if dataSet == 'cora':
            cora_content_file = "cora/cora.content"
            cora_cite_file = "cora/cora.cites"

            feat_data = []
            labels = []
            node_map = {}
            label_map = {}

            with open(cora_content_file) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    feat_data.append([float(x) for x in info[1:-1]])
                    node_map[info[0]] = i
                    if info[-1] not in label_map:
                        label_map[info[-1]] = len(label_map)
                    labels.append(label_map[info[-1]])
            feat_data = np.asarray(feat_data)
            labels = np.asarray(labels, dtype=np.int64)

            adj_lists = collections.defaultdict(set)
            with open(cora_cite_file) as fp:
                for line in fp:
                    info = line.strip().split()
                    paper1 = node_map[info[0]]
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)

            self.feat_data = feat_data
            self.labels = labels
            self.adj_lists = adj_lists

            test_indexs, val_indexs, train_indexs = self._split_data(len(feat_data))
            setattr(self, dataSet + '_test', test_indexs)
            setattr(self, dataSet + '_val', val_indexs)
            setattr(self, dataSet + '_train', train_indexs)

    def _split_data(self, num_nodes, test_split=3, val_split=6):
        rand_indices = np.random.permutation(num_nodes)
        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)
        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size + val_size)]
        train_indexs = rand_indices[(test_size + val_size):]
        return test_indexs, val_indexs, train_indexs
