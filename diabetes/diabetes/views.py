from rest_framework.views import APIView
import pickle
import pickle,pandas as pd
from rest_framework import status
from rest_framework.response import Response
from sklearn.metrics import accuracy_score



class GetdiabetesPrediction(APIView):
    def post(self,request):
        try:
            file = open(r'C:/Users/python-2/Desktop/New folder/diabetes/diabetes/Diabetes_model.pkl', 'rb')
            get_trained_model = pickle.load(file)
         
            user_data = {
                'Pregnancies':int(request.data.get("pregnancies",0)),
                'Glucose':int(request.data.get("glucose",0)),
                'BloodPressure':int(request.data.get("bloodPressure",0)),
                'SkinThickness':int(request.data.get("skinThickness",0)),
                'Insulin':int(request.data.get("insulin",0)),
                'BMI':int(request.data.get("BMI",0)),
                'DiabetesPedigreeFunction':int(request.data.get("diabetesPedigreeFunction",0)),
                'Age':int(request.data.get("Age",0)),
            }
            df = pd.DataFrame(user_data,index=[0])
            get_result = get_trained_model.predict(df)[0]
            get_dict = {
                "0":"You Can be Diabetic.",
                "1":"You Can't be Diabetic."
            }
            
            context = {
                "status":status.HTTP_200_OK,
                "success":True,
                "response":{
                    
                    "result":get_dict.get(str(get_result),None),
                    "user_data":user_data,
                    "accuracy":str(accuracy_score([get_result], get_trained_model.predict(df))*100)+'%'
                }
            }
            return Response(context,status=status.HTTP_200_OK)
        except Exception as exception:
            context = {
                "status":status.HTTP_400_BAD_REQUEST,
                "success":False,
                "response":str(exception)
            }
            return Response(context,status=status.HTTP_400_BAD_REQUEST)
           






