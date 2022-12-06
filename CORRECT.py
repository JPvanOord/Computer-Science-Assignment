import json
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from sklearn.utils import resample
import matplotlib.pyplot as plt
with open('tv.json') as data:
    Json = json.load(data)

#-----------------------------------------------------------------------------------------------------

#putting the data in a dataframe
df = []
for i in range(len(list(Json.values()))):
    if len(list(Json.values())[i]) == 1:
        df.append(list(Json.values())[i][0])
    else:
        for k in range(len(list(Json.values())[i])):
            df.append(list(Json.values())[i][k])
df = pd.DataFrame(df)

#making a list of most common brands 
brands = ["samsung", "supersonic","philips","sharp","nec","toshiba","hisense","sony", 
          "lg", "sanyo","coby","panasonic","sansui","vizio","viewsonic","sunbritetv",
          "haier", "optoma", "proscan","jvc", "pyle", "lg electronics", "sceptre",
          "magnavox","mitsubishi","compaq","hannspree","upstar","westinghouse", "rca"]

#making a list of all resolution types
resolution = ["1080p","720p","2160p"]

#lose featuresmap in order to make resampling work
df = df.drop(['featuresMap'], axis = 1)

#-----------------------------------------------------------------------------------------------------
#making train and test samples by bootstrapping
def bootstrapping(df):
   bootstrap = resample(df, replace=True, n_samples=len(df))
   train_sample = bootstrap.drop_duplicates()
   test_sample = pd.concat([df, train_sample]).drop_duplicates(keep=False)
   train_titles = train_sample["title"]
   train_modelid = train_sample["modelID"]
   train_shop = train_sample["shop"]
   test_titles = test_sample["title"]
   test_modelid = test_sample["modelID"]
   test_shop = test_sample["shop"]
   
   return train_titles, train_modelid, train_shop, test_titles, test_modelid, test_shop

titles = df['title']
#Cleaning the titles
def clean_data(titles): 
    for i in range(len(titles)):
        titles.iloc[i] = titles.iloc[i].lower()
        titles.iloc[i] = titles.iloc[i].replace('"',"inch")
        titles.iloc[i] = titles.iloc[i].replace("'"," ")
        titles.iloc[i] = titles.iloc[i].replace('/'," ")
        titles.iloc[i] = titles.iloc[i].replace('–'," ")
        titles.iloc[i] = titles.iloc[i].replace('-'," ")
        titles.iloc[i] = titles.iloc[i].replace(':'," ")
        titles.iloc[i] = titles.iloc[i].replace('&'," ")
        titles.iloc[i] = titles.iloc[i].replace('('," ")
        titles.iloc[i] = titles.iloc[i].replace('+'," ")
        titles.iloc[i] = titles.iloc[i].replace(')'," ")
        titles.iloc[i] = titles.iloc[i].replace('['," ")
        titles.iloc[i] = titles.iloc[i].replace(']'," ")
        titles.iloc[i] = titles.iloc[i].replace(' hz',"hz")
        titles.iloc[i] = titles.iloc[i].replace(' hz',"hz")
        titles.iloc[i] = titles.iloc[i].replace("diag.","diagonal ")
        titles.iloc[i] = titles.iloc[i].replace("diag ","diagonal ")
        titles.iloc[i] = titles.iloc[i].replace('diagonally','diagonal')
        titles.iloc[i] = titles.iloc[i].replace('digaonal','diagonal')
        titles.iloc[i] = titles.iloc[i].replace('newegg.com'," ")
        titles.iloc[i] = titles.iloc[i].replace('thenerds.net'," ")
        titles.iloc[i] = titles.iloc[i].replace('best buy'," ")
        titles.iloc[i] = titles.iloc[i].replace('.'," ")
        titles.iloc[i] = titles.iloc[i].replace(','," ")
        titles.iloc[i] = titles.iloc[i].replace("”"," ")
        titles.iloc[i] = titles.iloc[i].replace('  '," ")
        titles.iloc[i] = titles.iloc[i].replace('  '," ")
        titles.iloc[i] = titles.iloc[i].replace('  '," ")
        titles.iloc[i] = titles.iloc[i].replace('  '," ")
        titles.iloc[i] = titles.iloc[i].replace(' inch',"inch")
        titles.iloc[i] = titles.iloc[i].replace('refurbished',"")
        titles.iloc[i] = " " + titles.iloc[i] + " "
    #making vector of brands and resolution to compare later    
    brandvector = np.zeros(len(titles))
    for i in range(len(titles)):
        for j in range(len(brands)):
            if (" " + brands[j] + " ") in titles.iloc[i]:
                brandvector[i] = j+1
                break
    resolutionvector = np.zeros(len(titles))
    for i in range(len(titles)):
        for j in range(len(resolution)):
            if (" " + resolution[j] + " ") in titles.iloc[i]:
                resolutionvector[i] = j+1
                
    return titles, brandvector, resolutionvector


#-----------------------------------------------------------------------------------------------------
def binary_matrix(titles):
    #making a vector of all titlewords
    titlewords = ["" for x in range(30000)]
    count = 0
    for i in range(len(titles)):
        current_title_words = titles.iloc[i].split()
        for j in range(len(current_title_words)):
            titlewords[j+count] = current_title_words[j]
        count = count + len(current_title_words)
        
    #-----------------------------------------------------------------------------------------------------    

    #removing all duplicate values from the titlewords vector
    titlewords = set(titlewords)
    seen = set()
    result = []
    for item in titlewords:
        if item not in seen:
            seen.add(item)
            result.append(item)
            
    #-----------------------------------------------------------------------------------------------------        
      
    #removing empty element of titlewords vector
    titlewords.remove("") 
    titlewords = list(titlewords)


    #creating matrix consisting of binary vectors, spaces are for recognising whole "words"
    binarymatrix = np.zeros((len(titlewords),len(titles)))
    for i in range(len(titlewords)):
        for j in range(len(titles)):
                       if (" "+ titlewords[i]+" ") in titles.iloc[j]:
                           binarymatrix[i,j] = 1
                           
    return binarymatrix, titlewords

#-----------------------------------------------------------------------------------------------------
def candidate_pairs(bands , rows, binarymatrix, titlewords, titles):
    permutations = int(bands*rows)
    signaturematrix = np.zeros((permutations,len(titles)),dtype=np.int64)  
    evolve = 0
    #generating signaturematrix
    for j in range(permutations):
        np.random.shuffle(binarymatrix)
        evolve = j 
        for z in range(len(titles)):
            a = binarymatrix[:,z]
            for i in range(len(titlewords)):
                if a[i] == 1 :
                    signaturematrix[evolve,z] = i
                    break
                
    #-----------------------------------------------------------------------------------------------------            
    
    bandmatrix = np.zeros((bands,len(titles)))
    z = np.arange(0,permutations,rows)
    bandsDict = {}
    #dividing bands in the dictionary
    for i in range(len((z))-1):
        bandsDict["band{0}".format(i)]=signaturematrix[z[i]:z[i+1]][:]
    bandsDict["band{0}".format(bands-1)]=signaturematrix[z[bands-1]:][:]

    #concatenate the column values
    for i in range(bands):
        for j in range(len(titles)):
           merge1 = bandsDict['band'+str(i)][:,j] 
           merge2 = [str(int) for int in merge1] 
           merge3 = ''.join(merge2) 
           bandmatrix[i,j] = int(merge3)
           
            
    #making candidate pairs
    #whenever two columns (a,b) have one band which is the same for the same row 
    #make element (a,b) equal to 1
    #Candidate pairs are all ones in the candidate_pair matrix   
    countpair = 0
    candidate_pair = np.zeros((len(titles),len(titles)),dtype=np.int64)
    for j in range(len(titles)):
        for z in range(j+1,len(titles)-1):
            for k in range(bands):
                if (bandmatrix[k,j] == bandmatrix[k,z]):
                    candidate_pair[z,j] = 1
                    candidate_pair[j,z] = 1
                    countpair = countpair + 1
                    break
                
    return candidate_pair, countpair
 
           
#-----------------------------------------------------------------------------------------------------            
#general function to calculte the jaccard similarity
def jaccard_binary(x,y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity



#-----------------------------------------------------------------------------------------------------            
def dissimilarity(candidate_pair, binarymatrix, titles, brandvector, shop, resolutionvector): 
    #filling the dissimilarity matrix, whenever a pair isnt a candidate pair then set
    #Dissimilarity to a high number (1000)
    #Whenever a pair is a candidate pair take the jaccard dissimilarity = 1 - jaccardsimilarity     
    #also require the products to be from a different shop, have the same brand and resolution
    dissmatrix = np.zeros((len(titles),len(titles)),dtype=float)
    for j in range(len(titles)):
        for i in range(j+1,len(titles)):
            if (candidate_pair[i,j] == 1 and shop.iloc[j] != shop.iloc[i] and brandvector[j] == brandvector[i]
                and resolutionvector[i] == resolutionvector[j]):
                dissmatrix[i,j] = (1-jaccard_binary(binarymatrix[:,i],binarymatrix[:,j]))
                dissmatrix[j,i] = (1-jaccard_binary(binarymatrix[:,i],binarymatrix[:,j]))
            else:
                dissmatrix[i,j] = 1000
                dissmatrix[j,i] = 1000
    return dissmatrix



def results(dissmatrix, threshold, modelid, countpair):
    clustering = AgglomerativeClustering(affinity='precomputed', linkage='complete', distance_threshold = threshold, n_clusters=None)
    clustering = clustering.fit(dissmatrix)
    labels = clustering.labels_
    #making a list of predicted duplicates
    pred_dubs = []
    for i in range(0, clustering.n_clusters_):
        clusprods = np.where(labels == i)[0]
        if (len(clusprods)>1):
            pred_dubs.extend(list(combinations(clusprods, 2)))
    #list of real duplicates        
    real_dubs = []
    for i in modelid:
        if i not in real_dubs:
            dubs = np.where(modelid == i)[0]
            if (len(dubs)>1):
                real_dubs.extend(list(combinations(dubs, 2)))

    set_real = set(real_dubs) #removing duplicates
    real_dubs = list(set_real) 
    comp_total = (len(modelid)*(len(modelid)-1))/2 #total comparisons
    

    tp = 0
    fp = 0
    for i in range(0,len(pred_dubs)):
        if pred_dubs[i] in real_dubs:
            tp += 1
        else:
            fp +=1
        
    fn = len(real_dubs)-tp


    pq = tp/(countpair)
    pc = tp / (tp + fn)
    f1 = tp / (tp + (fp+fn)/2)
    f1star = (2 * pq * pc) / (pq + pc)
    comparison_percentage = countpair / comp_total 
    
    return pq, pc, f1, f1star, comparison_percentage


#determining results
def sampling(df):
    train_titles, train_modelid, train_shop, test_titles, test_modelid, test_shop = bootstrapping(df)
    test_titles, brandvector, resolutionvector = clean_data(test_titles)
    binarymatrix, titlewords = binary_matrix(test_titles)
    return binarymatrix, titlewords, test_titles, brandvector, test_shop, resolutionvector, test_modelid
    

def final1(threshold, bands, rows, binarymatrix, titlewords, test_titles,test_modelid, brandvector, test_shop, resolutionvector):   
    candidate_pair, countpair = candidate_pairs(bands , rows, binarymatrix, titlewords, test_titles)
    dissmatrix = dissimilarity(candidate_pair, binarymatrix, test_titles, brandvector, test_shop, resolutionvector)
    pq, pc, f1, f1star, comparison_percentage = results(dissmatrix, threshold, test_modelid, countpair)
    return pq, pc, f1, f1star, comparison_percentage


binarymatrix1, titlewords1, test_titles1, brandvector1, test_shop1, resolutionvector1, test_modelid1 = sampling(df)
binarymatrix2, titlewords2, test_titles2, brandvector2, test_shop2, resolutionvector2, test_modelid2 = sampling(df)
binarymatrix3, titlewords3, test_titles3, brandvector3, test_shop3, resolutionvector3, test_modelid3 = sampling(df)
binarymatrix4, titlewords4, test_titles4, brandvector4, test_shop4, resolutionvector4, test_modelid4 = sampling(df)
binarymatrix5, titlewords5, test_titles5, brandvector5, test_shop5, resolutionvector5, test_modelid5 = sampling(df)


def stats(binarymatrix, titlewords, titles, brandvector, shop, resolutionvector, modelid):
    
    pq1, pc1, f1_1, f1star1, comparison_percentage1 = final1(0.5, 100, 1, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)
    pq2, pc2, f1_2, f1star2, comparison_percentage2 = final1(0.5, 50, 2, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)
    pq3, pc3, f1_3, f1star3, comparison_percentage3 = final1(0.5, 33, 3, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)
    pq4, pc4, f1_4, f1star4, comparison_percentage4 = final1(0.5, 25, 4, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)
    pq5, pc5, f1_5, f1star5, comparison_percentage5 = final1(0.5, 20, 5, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)
    pq6, pc6, f1_6, f1star6, comparison_percentage6 = final1(0.5, 17, 6, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)
    pq7, pc7, f1_7, f1star7, comparison_percentage7 = final1(0.5, 14, 7, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)
    pq8, pc8, f1_8, f1star8, comparison_percentage8 = final1(0.5, 12, 8, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)
    pq9, pc9, f1_9, f1star9, comparison_percentage9 = final1(0.5, 11, 9, binarymatrix, titlewords, titles, modelid, brandvector, shop, resolutionvector)

    
    
    pqvector = np.array([pq1, pq2, pq3, pq4, pq5, pq6, pq7, pq8, pq9])
    pcvector = np.array([pc1, pc2, pc3, pc4, pc5, pc6, pc7, pq8, pq9])
    f1vector = np.array([f1_1, f1_2, f1_3, f1_4, f1_5, f1_6, f1_7, f1_8, f1_9])
    f1starvector = np.array([f1star1, f1star2, f1star3, f1star4, f1star5, f1star6, f1star7, f1star8, f1star9])
    percentagevector = np.array([comparison_percentage1, comparison_percentage2, comparison_percentage3, comparison_percentage4, 
                                 comparison_percentage5, comparison_percentage6, comparison_percentage7, comparison_percentage8, comparison_percentage9])
    
    return pqvector, pcvector, f1vector, f1starvector, percentagevector
    
pqvector1, pcvector1, f1vector1, f1starvector1, percentagevector1 = stats(binarymatrix1, titlewords1, test_titles1, brandvector1, test_shop1, resolutionvector1, test_modelid1)    
pqvector2, pcvector2, f1vector2, f1starvector2, percentagevector2 = stats(binarymatrix2, titlewords2, test_titles2, brandvector2, test_shop2, resolutionvector2, test_modelid2)    
pqvector3, pcvector3, f1vector3, f1starvector3, percentagevector3 = stats(binarymatrix3, titlewords3, test_titles3, brandvector3, test_shop3, resolutionvector3, test_modelid3)    
pqvector4, pcvector4, f1vector4, f1starvector4, percentagevector4 = stats(binarymatrix4, titlewords4, test_titles4, brandvector4, test_shop4, resolutionvector4, test_modelid4)    
pqvector5, pcvector5, f1vector5, f1starvector5, percentagevector5 = stats(binarymatrix5, titlewords5, test_titles5, brandvector5, test_shop5, resolutionvector5, test_modelid5)    


pq_average = np.zeros(len(pqvector1))
pc_average = np.zeros(len(pqvector1))
f1_average = np.zeros(len(pqvector1))
f1star_average = np.zeros(len(pqvector1))
percentage_average = np.zeros(len(pqvector1))

for i in range(len(pqvector1)):
    pq_average[i] = (pqvector1[i] + pqvector2[i] + pqvector3[i] + pqvector4[i] + pqvector5[i])/5
    pc_average[i] = (pcvector1[i] + pcvector2[i] + pcvector3[i] + pcvector4[i] + pcvector5[i])/5
    f1_average[i] = (f1vector1[i] + f1vector2[i] + f1vector3[i] + f1vector4[i] + f1vector5[i])/5
    f1star_average[i] = (f1starvector1[i] + f1starvector2[i] + f1starvector3[i] + f1starvector4[i] + f1starvector5[i])/5
    percentage_average[i] = (percentagevector1[i] + percentagevector2[i] + percentagevector3[i] + percentagevector4[i] + percentagevector5[i])/5



plt.plot(percentage_average, f1_average)
plt.plot(percentage_average, pq_average)
plt.plot(percentage_average, pc_average)

################################################################################################
#plot apart from eachother
plt.figure(num = 0, dpi=120)
plt.plot(percentage_average, f1_average, color = 'b')
plt.xlabel("Fraction of comparisons")
plt.ylabel("F1-score")
################################################################################################
plt.figure(num = 0, dpi=120)
plt.plot(percentage_average, pq_average, color = 'g')
plt.xlabel("Fraction of comparisons")
plt.ylabel("Pair quality")
################################################################################################

plt.figure(num = 0, dpi=120)
plt.plot(percentage_average, pc_average, color = 'r')
plt.xlabel("Fraction of comparisons")
plt.ylabel("Pair completeness")
################################################################################################

plt.figure(num = 0, dpi=120)
plt.plot(percentage_average, f1star_average, color = 'black')
plt.xlabel("Fraction of comparisons")
plt.ylabel("F1*-score")
################################################################################################





#used this for determining optimal threshold value
def final2(bands, rows, threshold):
    train_titles, train_modelid, train_shop, test_titles, test_modelid, test_shop = bootstrapping(df)
    train_titles, brandvector, resolutionvector = clean_data(train_titles)
    binarymatrix, titlewords = binary_matrix(train_titles)
    candidate_pair, countpair = candidate_pairs(bands , rows, binarymatrix, titlewords, train_titles)
    dissmatrix = dissimilarity(candidate_pair, binarymatrix, train_titles, brandvector, train_shop, resolutionvector)
    tp, fp, fn, f1, f1star, comparison_percentage = results(dissmatrix, threshold, train_modelid, countpair)
    return tp, fp, fn, f1, f1star, comparison_percentage
    
   


def optimise(bands, rows, threshold):
    f1 = final1(bands, rows, threshold)[3]
    f2 = final1(bands, rows, threshold)[3]
    f3 = final1(bands, rows, threshold)[3]
    f4 = final1(bands, rows, threshold)[3]
    f5 = final1(bands, rows, threshold)[3]
    avg = (f1 + f2 + f3 + f4 + f5)/5
    
    return avg


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



















    
