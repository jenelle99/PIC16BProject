# PIC16B Project Proposal
## *Abstract*
As a group, we are planning on developing a platform that identifies mental health disorders according to symptoms provided by the patient. Our platform will help classify mental disorders and support the clinician with diagnosing the mental health disorder. Also, it may raise awareness and encourage people to seek help. We will work with a large dataset with multiple variables, which will be eventually evaluated by a machine learning algorithm.

## *Planned Deliverables*
We aim to create a machine learning application to help diagnose mental disorders. We will provide a machine learning model to develop an indicator or classifier to determine the mental health state of a person to support early detection. We will analyze the given data to finally make a code that shows data set of results. 

If our project leads to a "Full success", our model will be used to detect a mental disorder from several features or states of a person. In this case, since we plan to create a code for a data set, Jupyter Notebook will be an appropriate interface.

On the other hand, if we happen to come up with a "Partial success", we will be at least able to show several relationships between features; for example, how an willingness to seek help from psychologists is related to a satisfaction with life? In this case, we will deliver data analysis in code, therefore a Jupyter Notebook will be an appropriate interface.

## *Resources Required*
For this project we indend to use the dataset found [here](https://datasets.simula.no/depresjon). This dataset containins sensor data collected from patients suffering from depression. The dataset contains motor activity recordings of 23 unipolar and bipolar depressed patients and 32 healthy controls. The data was collected through wearable sensors measuring different parts of people's activity which relates to various mental health issues such as changes in mood, personality, inability to cope with daily problems or stress. 

## *Tools and Skills Required*
In order to perform the tasks necessary for identifying mental health disorders, we will have to first analyze the data and find an appropriate machine learning algorithm for the platform. To assess the necessary machine learning algorithm and work on the project, we will use various Python packages such as NumPy, Panda, and Scikit Learn, which we have previously used in PIC16A. Python packages like Seaborn and Matplotlib may become useful in data visualization as well. In addition to this, Tensorflow and Statsmodels packages, which we have not yet learned, may become relevant when classifying the mental health disorders according to the data input. 

## *What You Will Learn*
We believe we will learn numerous skills and techniques from our project. First, we will learn to clean and preprocess our data. Then we need to understand and analyze our data, so we will use analytical skills using numpy, pandas, and other packages in Python. Also, we will know what to improve from our failures at multiple points throughout the project. And finally, we will make a presentation to audience to explain and introduce what we have done throughout the quarter.

## *Risks*
We might run into several problems and risks when trying to achieve the full deliverable. One main problem that can come up is that the dataset may not include enough variables and data necessary for analyzing and identifying the disorder. On the other hand, mental disorders’ characteristics are similar to each other, which may complicate identifying and categorizing the disorder. 

## *Ethics*
When it comes to developing tools that aim to help people, it is important to consider potential biases or harms from these tools. In our case, unfortunately, some groups of people have the potential to benefit more than others from the existence of our algorithm. Our project will likely be able to diagnose people of European ancestry more than other groups of people. That is due to the fact that more data has been collected on European ancestry individuals, and therefore we can translate our findings to other populations, but the predictive power that we have on European ancestry individuals will not be the same as other individuals. When it comes to mental health, attitudes toward this subject vary among different ethnicities, cultures, and countries. Cultural teachings often influence beliefs about the origins and nature of mental illness, and shape attitudes towards the mentally ill, and that could affect patients’ readiness and willingness to seek treatment. Therefore, less information will be available about people of these cultures, which could potentially lead to a lower diagnostic accuracy among those populations. Unfortunately, that only further grows the health inequality that we already are facing. However, we aim to do our part in finding diverse sets of data while also keeping in mind the potential harms when our algorithm is used.

We believe that our project could make the world an overall better place, and that is because mental health is an important factor in our daily lives but it is mostly disregarded and an easy to use tool could potentially help people arrive at the right diagnosis, and therefore seek the right treatment. This project might help people who are unaware of their mental health to understand their mental state and to seek help. However, we recommend seeing a professional alongside using this tool due to the possible inaccuracy mentioned above.


## *Tentative Timeline*
  - **Week 6**: We anicipate to be able to demonstrate successful data aquisition and manipulation. 
  - **Week 8**: We will show some data analysis and an implemented machine learning model that works with a reasonable accuracy.
  - **Week 10**: We will have our high-level-performance machine learning model and our tool will be ready to use.
