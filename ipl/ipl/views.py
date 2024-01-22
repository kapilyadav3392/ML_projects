from rest_framework.views import APIView
import pickle,pandas as pd
from rest_framework import status
from rest_framework.response import Response



class IPLPredictAPI(APIView):
    def post(self,request):
        try:
            data = request.data
            print(data)
            
            batting_team = data['Batting_Team']
            bowling_team = data['Bowling_Team']
            selected_city = data['City']
            target = data['Target']
            score = data['Score']
            overs = data['Over_Completed']  
            wickets = data['Wickets_Out']


            runs_left = int(target) - int(score)
            balls_left = 120 - (int(overs)*6)
            wickets = 10 - int(wickets)
            crr = int(score)/int(overs)
            rrr = (runs_left*6)/balls_left

            input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

            model = pickle.load(open(r"C:/Users/python-2/Desktop/New folder/ipl_prediction/ipl/model_pipe.pkl",'rb'))
            
            result = model.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]

            result1 = batting_team + "- " + str(round(win*100)) + "%"
            result2 = bowling_team + "- " + str(round(loss*100)) + "%"

            context = {"status":status.HTTP_200_OK,
                        "success":True,
                    "result1" :result1, 
                    "result2" :result2
                    }

            return Response(context)
        
        except Exception as e:
            context = {"status":status.HTTP_400_BAD_REQUEST,
                        "success":False, 
                        "response":str(e)
                        }
            return Response(context)