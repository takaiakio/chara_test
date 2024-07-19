from django.shortcuts import render
from .forms import UploadFileForm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import urllib

def handle_uploaded_file(f):
    data = pd.read_csv(f)
    return data

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            data = handle_uploaded_file(request.FILES['file'])
            # 対応分析の実行
            row_labels = data.iloc[:, 0]
            col_labels = data.columns[1:]
            data_matrix = data.iloc[:, 1:].values
            
            # 標準化
            scaler = StandardScaler()
            data_matrix_std = scaler.fit_transform(data_matrix)
            
            # PCAの実行
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(data_matrix_std)
            
            # プロット
            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=principalComponents[:, 0], y=principalComponents[:, 1])
            for i, label in enumerate(row_labels):
                plt.text(principalComponents[i, 0], principalComponents[i, 1], label)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Correspondence Analysis')
            
            # グラフを保存して表示
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            string = base64.b64encode(buf.read()).decode('utf-8')
            uri = 'data:image/png;base64,' + urllib.parse.quote(string)
            
            return render(request, 'analysis/result.html', {'image': uri})
    else:
        form = UploadFileForm()
    return render(request, 'analysis/upload.html', {'form': form})

