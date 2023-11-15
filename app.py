# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import joblib
# # from sklearn.ensemble import ExtraTreesClassifier
# # from prediction import get_prediction, label_encoder

# # model = joblib.load(r'Model/extree.joblib')

# # st.set_page_config(page_title="Accident Severity Prediction App",
# #                    layout="wide")


# # #creating option list for dropdown menu
# # options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
# # options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

# # options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
# #        ' Industrial areas', 'School areas', '  Recreational areas',
# #        ' Outside rural areas', ' Hospital areas', '  Market areas',
# #        'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
# #        'Recreational areas']
       
# # options_cause = ['No distancing', 'Changing lane to the right',
# #        'Changing lane to the left', 'Driving carelessly',
# #        'No priority to vehicle', 'Moving Backward',
# #        'No priority to pedestrian', 'Other', 'Overtaking',
# #        'Driving under the influence of drugs', 'Driving to the left',
# #        'Getting off the vehicle improperly', 'Driving at high speed',
# #        'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
# #        'Unknown', 'Improper parking']

# # options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
# #        'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
# #        'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
# #        'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

# # options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']

# # options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
# #        'other', 'Double carriageway (median)', 'One way',
# #        'Two-way (divided with solid lines road marking)', 'Unknown']

# # features = ['hour','day_of_week','casualties','accident_cause','vehicles_involved','vehicle_type','driver_age','accident_area','driving_experience','lanes']

# # # For the heading
# # st.markdown("<h1 style='text-align:center;'>Road Traffic Accident Severity Prediction</h1>")

# # def main():
# #     with st.form("prediction_form"):
# #         st.subheader("Enter the input for the following features : ")

# #         hour = st.slider("Pickup Hour: ", 0,24,value=0,format="%d")
# #         day_of_week = st.selectbox("Day of the Week: ",options = options_day)
# #         accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
# #         vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
# #         driver_age = st.selectbox("Select Driver Age: ", options=options_age)
# #         accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)
# #         driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
# #         lanes = st.selectbox("Select Lanes: ", options=options_lanes)

# #         submit = st.form_submit_button("Predict")
    
# #     if submit:
# #         day_of_week = label_encoder(day_of_week, options_day)
# #         accident_cause = label_encoder(accident_cause, options_cause)
# #         vehicle_type = label_encoder(vehicle_type, options_vehicle_type)
# #         driver_age =  label_encoder(driver_age, options_age)
# #         accident_area =  label_encoder(accident_area, options_acc_area)
# #         driving_experience = label_encoder(driving_experience, options_driver_exp) 
# #         lanes = label_encoder(lanes, options_lanes)

    
# #         data = np.array([hour,day_of_week,accident_cause, 
# #                             vehicle_type,driver_age,accident_area,driving_experience,lanes]).reshape(1,-1)
        
# #         pred = get_prediction(data=data, model=model)

# #         st.write(f"The predicted severity is:  {pred[0]}")

# # if __name__ == '__main__':
# #     main()
    
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from prediction import get_prediction, label_encoder

model = joblib.load(r'Model/model_subset.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_driving_exp = ['1-2yr' ,'Above 10yr', '5-10yr', '2-5yr' ,"nan" ,'No Licence' ,'Below 1yr','unknown']
options_road_surfacce = ['Dry' ,'Wet or damp' ,'Snow' ,'Flood over 3cm. deep'] 
options_types_of_junction = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape',
 'X Shape' ] 
options_casualty_sex = ['na', 'Male', 'Female'] 
options_driver_age_band=['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown'] 
options_light_conditions = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
 'Darkness - lights unlit'] 





features = ['Minutes','Day_of_week','Driving_experience','Road_surface_conditions','Types_of_Junction','Sex_of_casuality','Age_band_of_driver','Number_of_vehicles_involved','Number_of_casualities','Light_conditions']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        Minutes = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        Driving_experience = st.selectbox("Select Accident Cause: ", options=options_driving_exp)
        surface_conditions = st.selectbox("Select Surface COnditions: ", options=  options_road_surfacce)
        juction_type = st.selectbox("Select the type of Junction", options=options_types_of_junction)
        casualty_sex = st.selectbox("Select VGender of the casuality: ", options=options_casualty_sex)
        driver_age = st.selectbox("Select Driver Age: ", options= options_driver_age_band)
        vehicles_involved = st.slider("Number of Vehicles Involved: ", 1, 7, value=0, format="%d")
        casualities_involved = st.slider("Number of Casualities Involved: ", 1, 7, value=0, format="%d")
        light_conditions = st.selectbox("Select Driving Experience: ", options=options_light_conditions)
    
        
        
        submit = st.form_submit_button("Predict")


    if submit:
        day_of_week = label_encoder(day_of_week, options_day)
        Driving_experience = label_encoder(Driving_experience,options_driving_exp)
        surface_conditions = label_encoder( surface_conditions,options_road_surfacce)
        juction_type = label_encoder(juction_type,options_types_of_junction)
        casualty_sex = label_encoder(casualty_sex,options_casualty_sex)
        driver_age = label_encoder(driver_age,options_driver_age_band)
        light_conditions = label_encoder(light_conditions,options_light_conditions)

        data = np.array([Minutes,day_of_week,Driving_experience,surface_conditions, 
                            juction_type,casualty_sex,driver_age,vehicles_involved,casualities_involved,light_conditions]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()



