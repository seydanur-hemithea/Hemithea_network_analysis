Hemithea: AI-Powered Network Analysis Engine
​Hemithea, karmaşık veri kümelerindeki ilişkisel yapıları matematiksel modellerle haritalandıran ve interaktif olarak görselleştiren bağımsız bir analiz platformudur. Proje, veri biliminin Graph Theory (Çizge Kuramı) ve Unsupervised Learning (Denetimsiz Öğrenme) dallarını birleştirir.
​🛠 Teknik Mimari ve Algoritmalar
​Bu bağımsız Streamlit uygulamasında kullanılan temel yapılar şunlardır:
​1. Çizge Kuramı ve Dinamik Görselleştirme (NetworkX & Plotly)
​Fruchterman-Reingold Algoritması (Force-Directed Placement): Düğümleri (nodes) fiziksel birer parçacık gibi hayal eden bu algoritma, birbirine bağlı olanları çekerken, bağlı olmayanları iter.
​Ferahlık Kontrolü (k-parameter): Yazılımda kullanıcıya sunulan "itme kuvveti" ayarı, bu algoritmanın denge noktasını belirler. Bu sayede yüzlerce düğümden oluşan karmaşık ağlar (örneğin "Efendi" veri seti), kullanıcının isteğine göre ferahlatılabilir.
​Merkeziyet Analizi (Degree Centrality): Ağdaki "odak noktalarını" bulmak için kullanılır. Her düğümün bağlantı sayısı hesaplanır ve Plotly üzerinde bu değer düğüm boyutuyla (node_size) eşleştirilir.
​2. Yapay Zeka ile Otomatik Kümeleme (AI Clustering)
​Uygulama, yüklenen verideki toplulukları (communities) manuel müdahaleye gerek duymadan tespit eder:
​K-Means Clustering: Düğümlerin ağ içindeki konumları ve bağlantı yoğunlukları birer özellik (feature) olarak alınır. AI, bu özellikleri kullanarak benzer karakterdeki düğümleri aynı renk grubuna dahil eder.
​StandardScaler: Farklı ölçekteki verileri (az bağlantılı - çok bağlantılı) yapay zekanın doğru yorumlayabilmesi için veriler ortalama 0 ve standart sapma 1 olacak şekilde optimize edilir.
​3. Esnek Veri Entegrasyonu
​Dinamik Veri Yükleme: Pandas kütüphanesi ile desteklenen yapı; kullanıcıdan gelen CSV veya TXT dosyalarını anında işler, sütun isimlerini temizler (strip, lower) ve analiz motoruna hazır hale getirir.
​Hibrit Veri Yönetimi: Proje, hem GitHub üzerindeki statik "Efendi" veri setini hem de kullanıcının yerel dosyalarını aynı motor üzerinden işleyebilecek modüler bir yapıya sahiptir.




​🌐 Hemithea: AI-Powered Network Analysis Engine
​Hemithea is an independent analysis platform that maps relational structures in complex datasets using mathematical models and interactive visualizations. The project integrates Graph Theory and Unsupervised Learning techniques to extract actionable insights from unstructured data.
​🛠 Technical Architecture and Algorithms
​The platform utilizes a modular structure designed for performance and scalability:
​1. Graph Theory and Dynamic Visualization (NetworkX & Plotly)
​Fruchterman-Reingold Algorithm (Force-Directed Placement): The platform models nodes as physical particles, balancing repulsive and attractive forces to visualize connections. The "Repulsion Factor" (k-parameter) allows users to optimize layouts, preventing node overlap in highly dense networks (such as the "Efendi" dataset).
​Centrality Analysis (Degree Centrality): The application calculates the importance of each node based on the number of connections. Node sizes (node_size) are dynamically adjusted based on these centrality metrics to highlight key influencers.
​2. AI-Driven Clustering (Unsupervised Learning)
​To automatically identify communities within networks without manual intervention:
​K-Means Clustering: The application treats node positions and connectivity density as features. AI automatically detects clusters and assigns color-coded groups, facilitating the identification of sub-communities.
​StandardScaler: Before clustering, data is normalized (mean=0, std=1) to ensure the AI model interprets varying connectivity levels accurately, preventing outliers from skewing the results.
​3. Flexible Data Integration
​Dynamic Data Ingestion: Powered by Pandas, the engine processes user-uploaded CSV/TXT files by automatically cleaning headers and mapping relationships, ensuring a seamless user experience.
​Hybrid Data Management: The project features a modular design that handles both static datasets hosted on GitHub (e.g., the "Efendi" project) and user-provided local files through a single processing pipeline.
