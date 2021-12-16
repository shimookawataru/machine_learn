from sklearn import datasets #scikit-learnのデータセットモジュールのインポート


digits = datasets.load_digits() #手書き数字のデータセットの読み込み
print(digits.images[0])