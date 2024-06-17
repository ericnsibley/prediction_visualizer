[deployed app](https://rick-ml.com)

# Prediction Visualizer 
This project demonstrates a comprehensive machine learning deployment workflow, from stochastic data prep and model training to containerizing a visualization application that performs predictions and deploying it to a remote server. The application features live predictions using a pre-trained model visualized in a 3D point cloud.

## Project Overview

1. **Automated Data Load Process**: In the place of where I would have an idempotent data load, I created a github action that runs a data generation script and commits the result back to the repository. This represents the output of a feature engineering process, so my data is already all numerical and normalized. There is a 'fraud' column with the target labels of 1 and 0 for fraud / not fraud. 
2. **Automated Model Training**: After the data load there is another github action that trains an xgboosted tree on this data and writes it back as a pickle file, committing it to the repository. 
3. **Containerized Visualization Application**: I made an application that pulls in data, projects it down to 3d with PCA, displays is as a point cloud, and colors the points according to the 'fraud/ not fraud' labels. On hovering over a point a callback is ran that finds the original point before PCA and runs it through the model for a prediction. The visualization application is containerized using Docker so I can build it locally for testing, and then deploy it to my Digital Ocean server and know it will run exactly the same. Because my server is cheap it doesn't have enough memory to build the app, so I build it in a github action, store it on github container registry, then pull from there on the build server.
4. **Deployment**: The application is deployed using a GitHub Action to a Digital Ocean server that I attached a domain to. There's an nginx server in front of the application to terminate SSL. 

