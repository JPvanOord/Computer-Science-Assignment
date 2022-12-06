Assignment for the MSc Econometrics course "Computer Science For Business Analytics (FEM21037)" from the Erasmus University of Rotterdam. The project takes a close look at duplicate detection using data from the file "TVs-all-merged.json" containing information of 1624 online advertisements for televisions. The data is pre-processed using Locality-Sensitive Hashing (LSH) to reduce the number of comparisons to be made. The Jaccard measure is used to compute the similarity between the titles of products and clustering based on a threshold value is applied for duplicate detection. 

The Python code in this github is structured as follows:

1. Putting data in a dataframe
2. Making list of most common brands and all resolutions
3. Funnction which makes bootstrapsamples
4. Function for cleaning the titles and making brand- and resolutionvector 
5. Function for making the binary matrix, element (i,j) = 1 if titleword i is in title j
6. Function for making a candidate pair matrix: LSH on a signaturematrix
7. Function for making a dissimilarity matrix based on jaccard similarity if pair is candidate, shop is different, brand and resolution are the same otherwise dissimilarity = 1000
8. Function for duplicate detection via clustering
9. Final results
