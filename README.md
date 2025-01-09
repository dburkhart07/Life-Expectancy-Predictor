# Life Expectancy Predictor

An analysis on the performance and optimization of various classification ML models.  
This project implements each of them, using several optimized parameters to predict someone's life expectancy (or if they will live for 2+ years) based on their current health and habits.  

Attached are both a Python file and an identical Jupyter Notebook, as well as the training and testing CSV file for it.

---

## Running

### Installing Dependencies
This project uses Nodejs

Clone the repository and begin intalling dependencies.

Both the frontend and backend can have then dependencies installed by running:
```bash
npm install
```
If this does not work, try making sure you have node installed/the right version

### MongoDB Configuration
This project uses a MongoDB database. To appropriately set up a connetion for the database follow these steps:
<ol>
  <li>Make sure you have MongoDB installed</li>
  <li>Sign into MongoDB Atlas and create a new cluster</li>
  <li>This will allow you to then create a new MongoDB database</li>
  <li>Get a connection string and add that in your backend .env file</li>
</ol>

### Authentication
Create a random string for the JWT authentication, as well as the cookies secret

### OpenAI Integration
This project uses the OpenAI API for chat integration. In order to use the OpenAI API, you will need to use your own personal OpenAI Secret Key

### End
After all of this is over, just go into both the frontend and backend and run:
```bash
npm start
```
