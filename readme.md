
Deployment notes: 
I set up SSL to be self-renewing but you have to create the first cert on the deployment host with 
`docker run -it --rm --name certbot -v "/etc/letsencrypt:/etc/letsencrypt" -v "/var/lib/letsencrypt:/var/lib/letsencrypt" certbot/certbot certonly --standalone -d rick-ml.com -d www.rick-ml.com`

# Prediction Visualizer 

This project demonstrates a comprehensive machine learning deployment workflow, from data prep and model training to containerizing a visualization application and deploying it to a remote server. The application includes live predictions using a pre-trained model visualized in a 3D point cloud.

## Project Overview

1. **Automated Data Load Process**: In the place of where I would have an idempotent data load I created a github action that runs a data generation script and commits the result back to the repository.
2. **Automated Model Training**: After the data load there is another github action that trains an xgboosted tree and stores it as a pickle file, committing it back to the repository. 
3. **Containerized Visualization Application**: I made an application that pulls in data, projects it down to 3d, displays is as a point cloud, and colors the points according to the 'fraud/not fraud' labels. The visualization application is containerized using Docker so I can build it locally for testing, and then deploy it to my Digital Ocean server and know it will run exactly the same. Because my server is cheap it doesn't have enough memory to build the app, so I build it in a github action, store it on github container registry, then pull from there on the build server.
4. **Deployment**: The application is deployed using a GitHub Action to a Digital Ocean server that I attached a domain to. There's an nginx server in front of the application to terminate SSL. 

