import pandas as pd
import numpy as np
import time
import argparse


class CART:
    class TreeNode:
        def __init__(self, x=None, feature=None, children={}):
            self.feature = feature
            self.x = x
            self.children = children

        # 在决策树上迭代查找
        def search(self, s):
            if self.x is None:
                isContinuous = True if s[self.feature].dtype == np.float64 else False
                if isContinuous:
                    feature = list(self.children.keys())[0]
                    divide = float((feature.split("#")[1]))
                    if s[self.feature].values[0] > divide:
                        return self.children[">#" + feature.split("#")[1]].search(s)
                    else:
                        return self.children["<#" + feature.split("#")[1]].search(s)
                else:
                    # 由于数据量过大，在测试数据中存在部分找不到结果的样本，故使用故障捕捉
                    try:
                        return self.children[s[self.feature].values[0]].search(s)
                    except:
                        print("error")
            else:
                return self.x

    # 计算信息熵
    def ent(self, d):
        f = dict(d.iloc[:, -1].value_counts())
        # 统计k类的样本个数
        tot = len(d)
        ans = 0
        for i in f.keys():
            ans -= f[i] / tot * np.log2(f[i] / tot)
        return ans

    # 计算信息增益或增益率
    def gain(self, d, feature):
        ent_d = self.ent(d)
        max_ans = 0.0
        divide = -1
        # 数据是连续值时使用二分法
        if d[feature].dtype == np.float64:
            d.sort_values(by=feature)
            # 二分法中使用相邻两个元素的中值作为划分点，为简便书写，此处直接通过下标来划分，最后再另行计算划分点
            for i in range(len(d)):
                # 计算二分类划分的熵
                temp_ent = 0.0
                IV_d = 0.0
                temp_d = d.iloc[:i]
                temp_ent += (i + 1) / len(d) * self.ent(temp_d)
                IV_d -= len(temp_d) / len(d) * np.log2(len(temp_d) / len(d))
                temp_d = d.iloc[i + 1:]
                temp_ent += (len(d) - i - 1) / len(d) * self.ent(temp_d)
                IV_d -= len(temp_d) / len(d) * np.log2(len(temp_d) / len(d))
                ans = ent_d - temp_ent
                if self.args.mode == 'C4.5':
                    ans = ans / IV_d
                if ans > max_ans:
                    max_ans = ans
                    divide = i
        # 数据是连续时直接进行划分
        else:
            type_list = d[feature].unique()
            temp_ent = 0.0
            IV_d = 0.0
            for i in type_list:
                temp_data = d[d[feature] == i]
                temp_ent += len(temp_data) / len(d) * self.ent(temp_data)
                IV_d -= len(temp_data) / len(d) * np.log2(len(temp_data) / len(d))
            max_ans = ent_d - temp_ent
            if self.args.mode == 'C4.5':
                max_ans = max_ans / IV_d
        return max_ans, divide

    # 创建决策树
    def treeGenerate(self, df, features):
        label = self.label
        root = self.TreeNode(x=None, feature=None, children={})
        isAllIdentical = True
        # D中样本在A上取值相同时
        for feature in features:
            if df[feature].nunique() != 1:
                isAllIdentical = False
                break
        # D中样本全属于同一类别时
        if df.iloc[:, -1].nunique() == 1:
            x = df.iloc[:, -1].unique()
            root.x = x
        elif not features or isAllIdentical:
            x = df.loc[:, label].value_counts().idxmax()
            root.x = x
        else:
            # 选取最优划分属性
            max_ans = -1e9
            best_feature = None
            best_divide = -1
            for feature in features:
                temp_ans, divide = self.gain(df, feature)
                if temp_ans > max_ans:
                    max_ans = temp_ans
                    best_feature = feature
                    best_divide = divide
            root.feature = best_feature
            if best_divide != -1:
                # 连续值通过二分进行划分
                divide_d = df[df[best_feature] > best_divide]
                if len(divide_d) == 0:
                    x = df.loc[:, label].value_counts().idxmax()
                    root.children[">#" + str(best_divide)] = self.TreeNode(x, best_feature)
                else:
                    next_features = features.copy()
                    # 这里注意要浅复制
                    next_features.remove(best_feature)
                    root.children[">#" + str(best_divide)] = self.treeGenerate(divide_d, next_features)
                divide_d = df[df[best_feature] <= best_divide]
                if len(divide_d) == 0:
                    x = df.loc[:, label].value_counts().idxmax()
                    root.children["<#" + str(best_divide)] = self.TreeNode(x)
                else:
                    next_features = features.copy()
                    # 这里注意要浅复制
                    next_features.remove(best_feature)
                    root.children["<#" + str(best_divide)] = self.treeGenerate(divide_d, next_features)
            else:
                best_features = df[best_feature].unique()
                for feature in best_features:
                    divide_d = df[df[best_feature] == feature]
                    if len(divide_d) == 0:
                        x = df.loc[:, label].value_counts.idxmax()
                        root.children[feature] = self.TreeNode(x)
                    else:
                        next_features = features.copy()
                        # 这里注意要浅复制
                        next_features.remove(best_feature)
                        root.children[feature] = self.treeGenerate(divide_d, next_features)
        return root

    def test(self):
        print("start test")
        testset = self.testset
        submission = []
        for i in range(len(testset)):
            s = testset.iloc[i:i + 1]
            ans = [i + 1, self.root.search(s)]
            submission.append(ans)
        pd.DataFrame(submission, columns=['id', 'Transported']).to_csv(self.args.save_path + 'ans.csv')

    def __init__(self, args):
        self.args = args
        start = time.time()
        if args.utf:
            self.trainset = pd.read_csv(args.train_path, encoding='utf-8')
            self.testset = pd.read_csv(args.test_path, encoding='utf-8')
        else:
            self.trainset = pd.read_csv(args.train_path)
            self.testset = pd.read_csv(args.test_path)
        self.features = list(self.trainset.columns[1:-1])
        for feature in self.features:
            self.testset[feature].astype(self.trainset[feature].dtype)
        self.label = self.trainset.columns[-1]
        # for i in range(len(self.testset)):
        #     s = self.testset.iloc[i:i+1]
        #     for feature in self.features:
        #         print(s[feature].values[0])
        print("generate the decision tree")
        self.root = self.treeGenerate(self.trainset, self.features)
        self.test()
        end = time.time()
        print(end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./表1.csv', help="输入训练数据地址")
    parser.add_argument('--utf', type=bool, default=True, help='是否使用utf-8打开文件')
    parser.add_argument('--test_path', type=str, default='./表1.csv', help="输入测试数据地址")
    parser.add_argument('--save_path', type=str, default='./', help="保存数据地址")
    parser.add_argument('--mode', type=str, default='C4.5', choices=['C4.5', 'ID3'], help="选择训练模式")
    args = parser.parse_args()
    print(args)
    a = CART(args)
