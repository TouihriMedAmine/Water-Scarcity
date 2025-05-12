from django import forms

class DroughtPredictionForm(forms.Form):
    Evap = forms.FileField(label='Evap', required=True)
    Rainf = forms.FileField(label='Rainf', required=True)
    RootMoist = forms.FileField(label='RootMoist', required=True)
    SoilM_0_10cm = forms.FileField(label='SoilM_0_10cm', required=True)
    TVeg = forms.FileField(label='TVeg', required=True)
