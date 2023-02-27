# Restaurant Recommendation in Toronto City.
### Contant
- Dataset Attributes
- User Attributes
- Restaurant Attributes.
- The flow of the recommendation process.
- If the user is an old customer.
- If the user is a new user.
- How to run the project?

### Dataset Attributes
 - Userid.
 - Restaurantid.
 - Rating given by the user to the following restaurant.

### User Attributes.
 - Date of Birth.
 - User latitude.
 - User longitude.
 - The type of food the following user prefer (Veg/Non Veg).

### Restaurant Attributes.
 - Availability of the restaurants on different time frames (0 - Not available , 1 - available).
  - '12 - 16'
  - '16 - 23:59
  - '8 - 11'
 - Type of Food (NonVeg,Veg,Both)
 - Type of cuisine

### The flow of the recommendation process.
 - we take user id and the maximum distance(in KM) of which he/she wants to find recommendations of the restaurant.
 - checking if the user is new user.
### If the user is an old customer
- Getting current time(IST note in time interval of 12AM to 8AM there will be no recommendations shown, as our sample of data which we considered does not contain such data.) 
- Finding the list of restaurant for which the user have rated before and the list of restaurants which he/she did not rate.
- Now we will predict(by custom DeepLearning model we build for collobrative filtering task) the ratings of the restaurants for which the user has not given    rating and arrange the list of restaurants based on there predicted ratings.
- Now in the list of restaurants for which users have already rated before, we need to find which of the restaurants user have actually liked as customers generally tend to go for same restaurant if they like the restaurant, we did this filtering by calculating deviation(rating given to the following restaurant - users average rating) of the users towards the following restaurant rating, if the deviation is positive, then we consider the following restaurant to recommend.
- Now we have applied filters for list of restaurants for which we have predicted rating of the restaurant which he did not visit and for list of restaurants which he liked 
  - First we are filtering these list of restaurants based on customers prefered distance range.
  - filtering by restaurent availability, checking by current time.
  - filtering by type of food the user prefered.
  - Then the recommend function will return 3 json data structure (over here we are only considering top 5 restaurants for rated and unrated restaurants)
    - 1st variable contains new restaurant recommendations, where the keys are recommended restaurant names and values are the restaurant attributes.
    - 2nd variable contains old restaurant which user may re-visit, where the keys are recommended restaurant names and values are the restaurant attributes.
    - 3rd variable contains the following user profile, where key is following userid and values are his attributes.
### If the user is a new user.
 - Getting current time(IST note in time interval of 12AM to 8AM there will be no recommendations shown, as our sample of data which we considered does not contain such data.) 
 - Getting coorindates of the new user.
 - Filtering all the restaurants by the distance specified by the user.
 - Finding the popular restaurants with the following distance.
 - filtering by restaurent availability by comparing to current time(IST).
 - Then recommendating top 5 restaurants from the above filtered list.

### How to run the project?
- clone the repo.
- Create virtual env and initialize the env(resource :- https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) 
- Install all the req packages from requirements.txt
 - Command :- pip install -r /path/to/requirements.txt
- Download the data from the following link and keep all .json file in database folder
  - https://drive.google.com/drive/folders/1-6dvChAwmik2FvfLhjMyZ10nuuogubDN?usp=sharing
- Download the model .pt file and keep the file in Models folder
  - https://drive.google.com/file/d/1rC15hsbvFdzUlLmMVmWCCyGnivEolFiJ/view?usp=sharing
- User the following command to run the flask app
 - Command :- python3 main.py
