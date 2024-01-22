from rest_framework.views import APIView
import pickle
from rest_framework import status
import numpy as np
from rest_framework.response import Response



class PridictCropAPI(APIView):
    def post(self,request):
        try:
            model = pickle.load(open(r'C:/Users/python-2/Desktop/New folder/crop_prediction/crop/model.pkl','rb'))
            sc = pickle.load(open(r'C:/Users/python-2/Desktop/New folder/crop_prediction/crop/standscaler.pkl','rb'))
            ms = pickle.load(open(r'C:/Users/python-2/Desktop/New folder/crop_prediction/crop/minmaxscaler.pkl','rb'))
            
            data = request.data
           
            N = data['Nitrogen']
            P = data['Phosporus']
            K = data['Potassium']
            temp = data['Temperature']
            humidity = data['Humidity']
            ph = data['Ph']
            rainfall = data['Rainfall']

            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            scaled_features = ms.transform(single_pred)
            final_features = sc.transform(scaled_features)
            prediction = model.predict(final_features)

            crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                        19: "Pigeonpeas", 20: "Kidney beans", 21: "Chickpea", 22: "Coffee"}

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "{} is the best crop to be cultivated right there".format(crop)
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

            context = {"status":status.HTTP_200_OK,
                        "success":True, 
                        "result" :result, 
                    
                    }
            return Response(context)
        

        except Exception as e:
            context = {"status":status.HTTP_400_BAD_REQUEST,
                        "success":False, 
                        "response":str(e)}
            return Response(context)