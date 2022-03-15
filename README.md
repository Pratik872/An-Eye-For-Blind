# An Eye For Blind Deployment Branch
This is a repository for deploying Eye For Blind project in Heroku Cloud. You will find the code in [main](https://github.com/Pratik872/An-Eye-For-Blind/tree/main) branch.

This project is deployed at heroku. Check [here](https://eye-for-blind.herokuapp.com/)

## Contents

- ### Folder Structure for Deployment : 

	Some files added for deployment to the existing files which are present in [main](https://github.com/Pratik872/An-Eye-For-Blind/tree/main) branch folder structure are:

	- Procfile : This is basically the file in which you have to mention the Flask application initialized and in which file it is initialised

	- requirements.txt : This file is already present in the 'main' branch folder structure. This file is required for dockerisation of the application and also to deploy it on heroku. While deploying the application on Heroku, this file gets triggered and all the dependencies are downloaded.

	- runtime.txt : This file is used to specify the python version which we want heroku to use.It will be detected automatically by python and all dependencies in 'requirements.txt' will be downloaded and installed using the pip of your python version. I have used this so that there will be no conflicts while deployment.It is not mandatory to use this file 