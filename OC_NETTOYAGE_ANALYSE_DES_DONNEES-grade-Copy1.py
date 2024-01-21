#!/usr/bin/env python
# coding: utf-8

# # ANALYSE DE DONNEE

# ## Objectif

# ### L'agence Santé publique France souhaite améliorer la base de données Open Food Facts en résolvant les erreurs et les valeurs manquantes. Elle envisage un système de suggestion pour faciliter la saisie des données.L'objectif est d'améliorer la précision des informations dans le domaine de la santé publique.
# 
# 

# ## Exploration

# - Variable Target    : nutrition_grade_fr
# 
# - Lignes et Colonnes : (320772, 162)
# 
# - Types de variables : *float64 : 106 (65,43%)
#                        *object  : 56  (34,57%)
# 
# - Decription         : Le jeu de donnée possedent plusieurs variables qui affiches des valeurs nutritionnel pour 100g,
#                        excepter pour la variable 'energy_100g' qui affiche en Kilojoule.
#                        Plusieurs variables contiennent des valeurs > 100g alors censé contenir 100g max et certaines des
#                        valeurs négatives
# 
# - Analyse des NANs   : Enormement de valeurs manquantes. Nous gardons les variables qui on moins de 50% de NANs.
# 
# - Outliers           : On remplace les Valeurs aberrantes par NANs qui nous servirons à l'étapes d'imputation.

# ## IMPORTATION

# In[3]:


#Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Importation du jeu de donnée
data = pd.read_csv('openfoodfacts.csv', sep='\t', low_memory=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)


# # 

# # EXPLORATION

# ###### 

# In[3]:


#Affichage du jeu de donnée
data.head(3)


# In[4]:


#Création d'une copie
df = data.copy()


# In[5]:


df.info(df.columns.to_list())


# In[6]:


df.dtypes.value_counts().plot.pie(autopct='%.2f%%')
plt.title('Modalités des variable qualitatives et quantatives')
pass


# In[7]:


df.describe()


# In[8]:


def valeurs_manquantes(df):
  
    # Identifier les valeurs manquantes
    missing_values = df.isnull().sum().sum()
    total_cells = df.size

    # Calculer le nombre de valeurs non manquantes
    non_missing_values = total_cells - missing_values

    # Calculer le pourcentage de valeurs non manquantes sur tout le jeu de données
    non_missing_percentage = (non_missing_values / total_cells) * 100

    # Calculer le pourcentage de valeurs manquantes sur tout le jeu de données
    missing_percentage = (missing_values / total_cells) * 100

    # Afficher les résultats
    print("Total des valeurs non manquantes :", non_missing_values, "soit", round(non_missing_percentage, 2), "%")
    print("Total des valeurs manquantes :", missing_values, "soit", round(missing_percentage, 2), "%")


    #Affichage heatmap des valeurs manquantes
    plt.figure(figsize=(20,10))
    ax = sns.heatmap(df.isna(), cbar=False)
    ax.set_title('')
    plt.show()
    pass

    #Mesurer le prourcentage de NANs dans nos différentes colonnes
    (df.isna().sum()/df.shape[0]).sort_values(ascending=True).map("{:.2%}".format)

    #Barplot des valeurs manquantes
    mno.bar(df, sort="ascending")
    pass

print(valeurs_manquantes(df))


# #### Sur le figure('heatmap'), tout ce qui est en blanc représante les valeurs manquantes. On observe la repartition des NANs en % avec enormement de valeurs manquantes.                                                                                                                                                      Nous gardons les variables qui on moins de 50% de NANs.

# # 

# # NETTOYAGE DES DONNEES

# #### 

# In[9]:


#ON supprime les lignes en double dans le dataframe `df` en se basant sur la colonne "code" et conserve 
#uniquement la première occurrence de chaque valeur unique.

#Affichage des doublons
print('le jeu posséde:',df.duplicated(subset=['code']).sum(),'doublons')

#Suppression des doublons
df=df.drop_duplicates(subset ="code",keep="first")


# In[10]:


#Suppression des colonnes a plus de 50% de valeurs manquantes
df = df[df.columns[df.isna().sum()/df.shape[0]<0.5]]


# In[11]:


#Suppression des colonnes inutiles
df = df.drop(['code','url','creator','created_t','created_datetime','last_modified_t',
               'last_modified_datetime','brands','brands_tags','brands','brands_tags',
               'states','states_tags','states_fr','countries','countries_tags','countries',
              'countries_tags','additives','additives_n','countries_fr','product_name','ingredients_text','serving_size',
              'ingredients_from_palm_oil_n','ingredients_that_may_be_from_palm_oil_n'], axis=1)


# In[12]:


#Transformation des valeurs minuscule en majuscule
df['nutrition_grade_fr'] = df['nutrition_grade_fr'].str.upper()


# In[13]:


#Remplacemt de tous les caractères '-' par '_'
df.columns = df.columns.str.replace('-', '_')


# #### Nous avons observé que plusieurs grades ne correspondent pas correctement au score nutritionnel dans nos données. Par conséquent, nous avons pris la décision de procéder à des ajustements afin de corriger cette situation.

# #### Voici les règles utiliser pour attribuer les grades en fonction des valeurs du score nutritionnel :
# 
# - Pour les valeurs du score nutritionnel comprises entre -15 et -2, attribuer le grade 'A' (vert).
# - Pour les valeurs du score nutritionnel comprises entre -1 et +3, attribuer le grade 'B' (vert clair).
# - Pour les valeurs du score nutritionnel comprises entre +4 et +11, attribuer le grade 'C' (jaune).
# - Pour les valeurs du score nutritionnel comprises entre +12 et +16, attribuer le grade 'D' (orange).
# - Pour les valeurs du score nutritionnel comprises entre +17 et +40, attribuer le grade 'E' (rouge).
# 
# En appliquant ces règles,nous pourraons associer les grades correspondants aux valeurs du score nutritionnel de l'ensemblee du dataset.

# In[19]:


# Appliquer la condition et mettre à jour la colonne cible
df['nutrition_grade_fr'] = df['nutrition_score_fr_100g'].apply(lambda x: 'A' if x <= -2 else 'B' if (-1 <= x < 4) else 'C' if (4 <= x < 12) else 'D' if (12 <= x < 17) else 'E' if (17 <= x < 41) else x)

#Suppression des lignes de la colonne 'nutrition_grade_fr' qui contient des valeurs manquantes
df = df.dropna(subset='nutrition_grade_fr')


# In[21]:


# Récupérer la liste des noms de colonnes du DataFrame
colonnes = df.columns.tolist()

# Déplacer la colonne "nutrition_grade_fr" à la fin de la liste des colonnes
colonnes.remove('nutrition_grade_fr')
colonnes.append('nutrition_grade_fr')

# Réorganiser les colonnes du DataFrame selon la nouvelle liste d'ordre des colonnes
df = df[colonnes]


# In[22]:


# Calculer le pourcentage de valeurs manquantes par variable
missing_percentages = (df.isnull().sum() / len(df)) * 100

# Créer un DataFrame contenant les pourcentages de valeurs manquantes
missing_df = pd.DataFrame({'Variable': missing_percentages.index, 'Pourcentage de valeurs manquantes': missing_percentages.values})

# Trier le DataFrame par ordre croissant des pourcentages de valeurs manquantes
missing_df = missing_df.sort_values('Pourcentage de valeurs manquantes', ascending=True)

# Créer le diagramme à barres des valeurs manquantes en pourcentage
plt.figure(figsize=(12, 8))
plt.barh(missing_df['Variable'], missing_df['Pourcentage de valeurs manquantes'], color='skyblue')

# Définir les étiquettes des axes
plt.xlabel('Pourcentage de valeurs manquantes')


# Afficher le diagramme à barres
plt.title('Pourcentage de valeurs manquantes par variable')
plt.show()

# Calculer le pourcentage total de valeurs manquantes
total_missing_percentage = (df.isnull().sum().sum() / df.size) * 100


# Calculer le nombre de valeurs manquantes par variable
missing_values = df.isnull().sum()

# Calculer le pourcentage de valeurs manquantes par variable
missing_percentages = (missing_values / len(df)) * 100

# Afficher les pourcentages de valeurs manquantes par variable
print("Pourcentage de valeurs manquantes par variable :")
for column, percentage in missing_percentages.items():
    print(f"{column}: {percentage:.2f}%")
    
# Afficher le pourcentage total de valeurs manquantes par rapport à avant
print("\nPourcentage total de valeurs manquantes:", total_missing_percentage, "%")


# In[23]:


plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)
pass


#                         On visualise beaucoup moins de blanc que sur le heatmap précedent

# In[24]:


# Nombre de lignes et de colonnes restantes
original_num_rows = data.shape[0]
original_num_cols = data.shape[1]
num_rows = df.shape[0]
num_cols = df.shape[1]

# Calcul du pourcentage de lignes et de colonnes restantes par rapport au jeu de données initial
percent_rows = (num_rows / original_num_rows) * 100
percent_cols = (num_cols / original_num_cols) * 100

# Affichage des résultats
print("Après la suppression des doublons et des colonnes inutiles, il nous reste", num_rows, "lignes et", num_cols, "colonnes.")
print("Cela représente", round(percent_rows, 2), "% des lignes et", round(percent_cols, 2), "% des colonnes du jeu de données initial.")


# # OUTLIERS

# # 

# In[25]:


df_outliers = df.copy()


# In[26]:


plt.figure(figsize=(15,5))
sns.boxplot(data=df_outliers.drop(['nutrition_grade_fr','energy_100g'],axis=1))
plt.xticks(rotation=80)
pass


# # 

# In[27]:


cols_100 = [col for col in df_outliers.columns if col.endswith('_100g') and col != 'energy_100g' and 
            col !='nutrition_score_fr_100g' and col !='nutrition_score_uk_100g']


def remove(x):
    if x < 0 or x > 100:
        return np.nan

    return x


for col in cols_100:
    df_outliers[col] = df_outliers[col].apply(lambda x : remove(x))


# Nous remplaçons par une valeur manquante les valeurs des variables qui sont inférieures à 0 ou supérieures à 100, à l'exception de 'energy_100g' qui est mesurée en kilojoules ainsi 'nutrition-score-fr_100g' et 'nutrition-score-uk_100g' dont les valeurs sont comprises entre -15 et +40.

# In[28]:


plt.figure(figsize=(15, 5))
sns.boxplot(data=df_outliers.drop(['energy_100g'], axis=1))
plt.xticks(rotation=80)
plt.title("Boxplot des variables (sauf energy_100g)")
pass


# #### On constate que les valeurs impossibles ont changé à np.nan

# # 

# In[29]:


def clean_data(df_outliers):

    # Identifier les valeurs manquantes
    missing_values = (df_outliers.isnull().sum()/df_outliers.shape[0]).sort_values(ascending=True).map("{:.2%}".format)
    #missing_values = df.isnull().sum()
    print("Valeurs manquantes par colonne :\n", missing_values)

    # Identifier les outliers avec le quartile
    Q1 = df_outliers.quantile(0.25)
    Q3 = df_outliers.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_outliers < (Q1 - 1.5 * IQR)) | (df_outliers > (Q3 + 1.5 * IQR)))

    print("Outliers par colonne :\n", outliers.sum())

print(clean_data(df_outliers))


# ## 

# #### Remplacement des valeurs aberrantes par np.nan afin de ne pas perdre de données, qui seront ensuite imputées.

# In[30]:


num_var = df_outliers.select_dtypes(include='float').drop([ 'nutrition_score_fr_100g','nutrition_score_uk_100g'],axis=1)

for col in num_var:
    Q1 = df_outliers[col].quantile(0.25)
    Q3 = df_outliers[col].quantile(0.75)
    IQR = Q3 - Q1


    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    def iqr_func(x, lower_limit, upper_limit):
        if x < lower_limit or x > upper_limit:
            return np.nan
        return x

    df_outliers[col] = df_outliers[col].apply(lambda x: iqr_func(x, lower, upper))


# In[31]:


print(clean_data(df_outliers))


# In[32]:


plt.figure(figsize=(15,5))
sns.boxplot(data=df_outliers.drop('energy_100g', axis=1))
plt.xticks(rotation=80);


# ### Les valeurs qui se situent juste au-dessus des boîtes à moustaches ne sont pas considérées comme des valeurs aberrantes, mais plutôt comme des valeurs extrêmes. Ces valeurs ne semblent pas indiquer une anomalie dans notre jeu de données, contrairement à celles qui sont particulièrement éloignées des autres, c'est-à-dire celles qui sont "isolées".

# ### 

# In[33]:


plt.figure(figsize=(15,5))
sns.boxplot(x=df_outliers['energy_100g'])
plt.xticks(rotation=80);


# In[34]:


for column in df_outliers.drop('nutrition_grade_fr',axis=1):
    max_value = df_outliers[column].max()
    min_value = df_outliers[column].min()

    print( column, "Max value of :", max_value, '\n', column,"Min value of:", min_value)


# # 

# ## IMPUTATION

# In[35]:


df_fillna = df_outliers.copy()
df_fillna_grade = df_outliers.copy()
df_knn = df_outliers.copy()


# 

# ## FILLNA

# ### 

#  La méthode `df.fillna(mean())` utilisée pour remplacer les valeurs manquantes peut s'avérer rapide, mais elle peut également introduire des biais dans les données. Il est important de prendre en compte le fait que lors de l'utilisation cette méthode, la moyenne est calculée sur l'ensemble des données sans tenir compte des catégories.                                                                                                                      
# 
# Par conséquent, si les valeurs manquantes sont différenciées selon des catégories spécifiques, l'imputation par la moyenne peut introduire des valeurs biaisées.
# 
# Il est recommandé d'adopter une approche plus sophistiquée pour l'imputation des valeurs manquantes,
# en tenant compte des caractéristiques spécifiques

# In[36]:


df_fillna = df_fillna.fillna(df_fillna.median())


# In[37]:


print(clean_data(df_fillna))


# ### 

# ## FILLNA / GRADE

# # 

# La méthode `df.fillna` en utilisant le regroupement par catégorie est aussi rapide que la première méthode d'imputation. 
# En utilisant l'imputation par catégorie, vous pouvez remplacer les valeurs manquantes dans chaque groupe ou catégorie du DF
# par une valeur appropriée, telle que la moyenne ou la médiane de ce groupe spécifique. Cette approche permet de
# conserver la cohérence et les caractéristiques propres à chaque catégorie lors de l'imputation des valeurs manquantes.
# 
# 

# In[38]:


#Imputation de chaque variables par grade avec la moyenne

for i in df_fillna_grade['nutrition_grade_fr'].unique():
    df_fillna_grade.loc[df_fillna_grade['nutrition_grade_fr']==i] = df_fillna_grade.loc[df_fillna_grade['nutrition_grade_fr']==i].fillna(df_fillna_grade.median())



# In[39]:


print(clean_data(df_fillna_grade))


# In[40]:


diff = df.shape[0] - df_fillna_grade.shape[0]
percentage_diff = round((diff / df.shape[0]) * 100, 2)

print('Nous avons perdus', diff, 'lignes soit (',percentage_diff,'%), depuis le jeu de départ' )


# 

# 
# #                                                      KNNImputer

# #### Le KNNImputer est un outil utilisé pour l'imputation des valeurs manquantes dans un ensemble de données en utilisant l'algorithme des k plus proches voisins (k-nearest neighbors). Il s'agit d'une méthode d'imputation basée sur les valeurs similaires des autres échantillons dans l'ensemble de données.

# #### Avant de commencer l'imputation, nous allons rechercher la valeur optimale de K (K_best_neighbors) qui sera utilisée pour le KNNImputer. Étant donné que notre variable cible est le "nutrition grade", le modèle le plus adapté est le KNeighborsClassifier. Cependant, ce modèle ne peut pas traiter directement les valeurs manquantes sous forme de NaN. Par conséquent, nous devrons les supprimer ou les remplacer par la moyenne des autres valeurs.

# ### KNeighborsClassifier

# ###### remplacement des NANs par la median

# In[41]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score


# ###### Suppression des NANs

# In[42]:


# Charger les données
X = df_knn.drop('nutrition_grade_fr', axis=1)
y = df_knn['nutrition_grade_fr']

# Éliminer les lignes contenant des valeurs manquantes
X_cleaned = X.dropna()
y_cleaned = y[X.index.isin(X_cleaned.index)]

# Recherche du meilleur k
k_values = range(2, 10)
best_accuracy = 0
best_k = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k )
    scores = cross_val_score(knn, X_cleaned, y_cleaned, cv=5, scoring='accuracy')
    score_accu = np.mean(scores)
    
    if score_accu > best_accuracy:
        best_accuracy = score_accu
        best_k = k

print("Meilleur k :", best_k)
print("Précision correspondante :", best_accuracy)


# ###### On peut voir que sur les 2methodes on obtient le 'Meilleur k à 3' avec une bonne précision de plus de 0,88

# ### KNNImputer 

# #####  Le KNNImputer ne peut pas traiter directement des valeurs non numériques. Je vais donc commencer par remplacer les valeurs catégoriques par des valeurs numériques.
# 

# In[43]:


#Remplacement des valeurs catégoriels en valeurs numériques
df_knn['nutrition_grade_fr'] = df_knn['nutrition_grade_fr'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4})


# In[44]:


#Importation de la bibliothèque
from sklearn.impute import KNNImputer

#création d'une instance de la classe KNNImputer avec n_neighbors=3
imputer = KNNImputer(n_neighbors=best_k)  

#Création d'un nouveau DataFrame nommé df_imputed en appliquant le KNNImputer aux données contenues dans df_knn.
df_imputed = pd.DataFrame(imputer.fit_transform(df_knn), columns=df_knn.columns)


# In[45]:


#On arrondi à zéro décimale les valeurs de la cible
df_imputed.nutrition_grade_fr = round(df_imputed.nutrition_grade_fr, 0)


# In[46]:


df_imputed.head()


# In[47]:


df_imputed['nutrition_grade_fr'] = df_imputed['nutrition_grade_fr'].replace({ 0:'A',  1:'B',  2:'C', 3:'D',  4:'E'})


# In[48]:


print(clean_data(df_imputed))


# # Aprés avoir nettoyer les données, d'avoir mis en np.nan les outliers puis les avoir imputer, je commencais mon analyse.

# ## TARGET

# In[49]:


# Trier le dataframe par ordre alphabétique de la variable 'nutrition_grade_fr'
df_sorted = df_imputed.sort_values('nutrition_grade_fr')

# Tracer le graphique de comptage avec les catégories triées
ax = sns.countplot(x=df_sorted['nutrition_grade_fr'])

# Calculer les pourcentages
total = len(df_sorted['nutrition_grade_fr'])
for p in ax.patches:
    percentage = '{:.2f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center')

# Ajouter un titre au graphique
plt.title('Distribution de la variable nutrition_grade')

# Afficher le graphique
plt.show()


# In[50]:


# Nombre de données par nutrition_grade_fr :

df_imputed.groupby('nutrition_grade_fr').mean()


# In[51]:


# Tri des catégories par ordre alphabétique
order = sorted(df_imputed['nutrition_grade_fr'].unique())

# Tracé des graphiques en boîte
for i in df_imputed.select_dtypes(include='float'):
    fig, axs = plt.subplots(1, 1, figsize=(15, 5))
    sns.boxplot(x=df_imputed['nutrition_grade_fr'], y=df_imputed[i], order=order, ax=axs, color='#16E4CA')
    plt.title(i)

# Affichage des graphiques
plt.show()


# In[52]:


numeric_columns = df_imputed.select_dtypes(include=['float']).columns

for column in numeric_columns:
    plt.figure(figsize=(10, 2))
    sns.set(font_scale=1)
    sns.kdeplot(df_imputed[column])
    chaine = 'Distribution de : ' + column
    plt.title(chaine)
    plt.xlabel(column)
    plt.show()


# In[77]:


numeric_columns = df_imputed.select_dtypes(include=['float']).columns
categorical_variable = 'nutrition_grade_fr'

# Trier les catégories par ordre alphabétique
sorted_categories = sorted(df_imputed[categorical_variable].unique())

for column in numeric_columns:
    plt.figure(figsize=(10, 3))
    for category in sorted_categories:
        sns.kdeplot(data=df_imputed[df_imputed[categorical_variable] == category][column], label=category, fill=True)
    plt.title('Relation entre : ' + column + ' et ' + categorical_variable)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend(title=categorical_variable)
    plt.show()


# In[ ]:





# In[53]:


variables = df_imputed.drop('nutrition_grade_fr', axis=1).columns

sorted_grades = sorted(df_imputed['nutrition_grade_fr'].unique())  

# Définition d'une palette de couleurs personnalisée avec une couleur différente pour chaque colonne
colors = sns.color_palette('Set1', len(variables))

for variable, color in zip(variables, colors):
    g = sns.FacetGrid(df_imputed, col='nutrition_grade_fr', col_order=sorted_grades)
    g.map(sns.histplot, variable, color=color)
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(f'Distribution de {variable} selon les grades nutritionnels')
    plt.show()



# ##### Les distributions sont asymétriques.                     On voit que pour la majorité des aliments avec un grade A que moins il y a de gramme dans c plus la densité est élevé Pour la majorité des variables ayant un grade A on voit que moins les produits gramme plus leur densité est forte. Le grade A se concentre majoritairement entre 0 et 10 puis augmente au fur et a mesure que le grade augment.

# # 

# In[55]:


plt.figure(figsize=(20, 20))
sns.pairplot(df_imputed, corner=True)
plt.show()
pass


# In[56]:


df_imputed['nutrition_grade_fr'] = df_imputed['nutrition_grade_fr'].replace({ 0:'A',  1:'B',  2:'C', 3:'D',  4:'E'})


# In[1]:


plt.figure(figsize=(20, 20))
mask = np.triu(np.ones_like(df_imputed.corr()))
sns.heatmap(df_imputed.corr(), cmap='RdBu' ,annot=True,mask=mask, linewidths=.5)
pass


# #### Corrélation positif à 73% entre "saturated_fat" et "nutriscore_fr" elles suivent une même tendance contrairement au "saturated_fat" et "salt_100g".  Il y a une correlation tres forte entre sodium et sel qui est de 1, donc concerver les 2 napporte rien a la comprenesion de la variance du modele

# # 

# In[58]:


df_imputed.drop('sodium_100g', axis=1, inplace=True)


# ## ACP

# In[59]:


df_imputed['nutrition_grade_fr'] = df_imputed['nutrition_grade_fr'].replace({ 0:'A',  1:'B',  2:'C', 3:'D',  4:'E'})


# In[60]:


target = df_imputed['nutrition_grade_fr']
df_acp = df_imputed.drop('nutrition_grade_fr', axis=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
Z = sc.fit_transform(df_acp)
Z


# In[61]:


from sklearn.decomposition import PCA

pca = PCA()

# Contient les coordonnées de l'ACP sur les lignes.
Coord = pca.fit_transform(Z)


# In[62]:


print('Les valeurs propres sont :', pca.explained_variance_)

# plt.figure(figsize=(10, 10))
plt.plot(range(0, 10), pca.explained_variance_)
plt.xlabel('Nombre de facteurs')
plt.ylabel('Valeurs propres');


# In[63]:


# Les ratios sont calculés à partir de pca.explained_variance_ratio_
ratios = pca.explained_variance_ratio_

# Graphe de la somme cumulative de la variance expliquée
cumulative_sum = np.cumsum(ratios)
plt.plot(np.arange(1, 11), cumulative_sum)
plt.xlabel('Numéro du facteur')
plt.ylabel('Somme cumulative')
plt.title('Somme cumulative de la variance expliquée')

# Ajout des annotations des pourcentages
for i, cumsum in enumerate(cumulative_sum):
    plt.annotate(f'{cumsum*100:.2f}%', (i+1, cumulative_sum[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Affichage du graphe
plt.show()


# In[65]:


print('Les ratios sont :', pca.explained_variance_ratio_)

# Calcul de la somme cumulative des ratios
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Création du graphe en barres avec les pourcentages annotés
fig, ax1 = plt.subplots()

ax1.bar(np.arange(1, 11), pca.explained_variance_ratio_ * 100, color='b')
ax1.set_xlabel('Numéro du facteur')
ax1.set_ylabel('Pourcentage de variance expliquée', color='b')
ax1.set_title('Graphe de la somme cumulative de la variance expliquée')

# Annotation des pourcentages pour les barres
for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    ax1.annotate(f'{round(var_ratio * 100, 2)}%', (i+1, var_ratio * 100), textcoords="offset points", xytext=(0, 5), ha='center', color='b')


plt.show()


# In[67]:


# Camembert de la répartition de la part de variance expliquée par chaque axe.

L1 = list(pca.explained_variance_ratio_[0:5])
L1.append(sum(pca.explained_variance_ratio_[5:]))

plt.figure(figsize=(7,7))
plt.pie(L1, labels=['PC1', 'PC2', 'PC3', 'PC4', 'PC5',  'Autres'],
        autopct='%1.3f%%')
plt.title("Répartition de la part de variance expliquée par chaque axe")
plt.show()




# ### On remarque alors que la part de variance expliquée en fonction du nombre de facteurs atteint les 60% pour 2 composantes et jusqu'à plus de 80% pour 4 composantes.

# # 

# In[68]:


Comp_PCA = pd.DataFrame({'PC1': pca.components_[:, 0], 'PC2': pca.components_[:, 1]})

plt.figure(figsize=(20, 20))

sns.heatmap(Comp_PCA, annot=True, cmap='viridis');

PCA_mat = pd.DataFrame({'AXE 1': Coord[:, 0], 'AXE 2': Coord[:, 1], 'target': target})


# In[69]:


racine_valeurs_propres = np.sqrt(pca.explained_variance_)
corvar = np.zeros((10, 10))
for k in range(10):
    corvar[:, k] = pca.components_[:, k] * racine_valeurs_propres[k]

# Délimitation de la figure
fig, axes = plt.subplots(figsize=(10, 10))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)

# Affichage des variables
for j in range(10):
    plt.annotate(df_acp.columns[j], (corvar[j, 0], corvar[j, 1]), color='#091158')
    plt.arrow(0, 0, corvar[j, 0]*0.6, corvar[j, 1]*0.9, alpha=0.6, head_width=0.03, color='b')

# Ajout des axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

# Cercle et légendes
cercle = plt.Circle((0, 0), 1, color='#16E4CA', fill=False)
axes.add_artist(cercle)
plt.xlabel('Axe 1')
plt.title('Visualisation des variables sur le plan factoriel')
plt.show()


#       Les variables les plus corrélées à l'axe 1 sont :
#                                   négativement : 'nutrition-score-fr_100g', 'proteins_100g
#                                   positivement : 'energy_100g','fiber_100g'
#                                   
#       Les variables les plus corrélées à l'axe 2 sont :
#                                   négativement : 'carbohydrates_100g','salt_100g', 'proteins_100g'
#                                   positivement : 'nutrition-score-fr_100g', 'energy_100g'

# In[70]:


plt.figure(figsize=(10, 10))
PCA_mat['target'] = PCA_mat['target'].astype(str)  # Convertir la variable cible en type chaîne de caractères
PCA_mat = PCA_mat.sort_values('target')  # Trier les données en fonction de la variable cible
sns.scatterplot(x='AXE 1', y='AXE 2', hue='target', data=PCA_mat)
plt.xlabel('Axe 1')
plt.ylabel('Axe 2')
plt.title("Visualisation des données dans le plan factoriel (Component 1 vs Component 2)")

plt.show()


# #### Nous voyons ici que les cinq classes de notre variable target(a,b,c,d,e) semblent être distinctes. On a donc une bonne représentation de nos variables par l'ACP

# # 

# ##  TEST ANOVA 

# ###### 

#                   Hypothèse nulle (H0) : Les moyennes des groupes sont égales (il n'y a pas d'association significative
#                   entre la variable catégorielle et la variable numérique).
# 
#                   Hypothèse alternative (H1) : Au moins une moyenne des groupes est différente (il y a une association
#                   significative entre la variable catégorielle et la variable numérique).
#  

# 

# In[79]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

# Variables catégorielle et numériques
cat_var = df_imputed['nutrition_grade_fr']
numeric_vars = df_imputed.drop('nutrition_grade_fr', axis=1)  

# Boucle pour l'ANOVA
results = {}
for col in numeric_vars.columns:
    # ANOVA
    model = ols('{} ~ C({})'.format(col, cat_var.name), data=df_imputed).fit()
    anova_table = sm.stats.anova_lm(model)
    p_value_anova = anova_table['PR(>F)'][0]
    
    # Stocker les résultats
    results[col] = {'p_value_anova': p_value_anova}

# Afficher les résultats
for col, values in results.items():
    print(f"Variable : {col} & {cat_var.name}")
    print(f"                                   p-value ANOVA : {values['p_value_anova']}")
    print()


# #### Les résultats du test ANOVA indiquent que toutes les variables ont une valeur de p inférieure à 0.05. Cela suggère qu'il existe une corrélation significative entre ces variables et la variable cible. 

# ### 

# In[78]:


diff = data.shape[0] - df_imputed.shape[0]
percentage_diff = round((diff / data.shape[0]) * 100, 2)

print('Nous avons perdus', diff, 'lignes soit (',percentage_diff,'%), depuis le jeu de départ' )


# In[ ]:




