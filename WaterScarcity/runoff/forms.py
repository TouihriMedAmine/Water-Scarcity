from django import forms

class RunoffPredictionForm(forms.Form):
    image_day1 = forms.ImageField(label='Day 1 Image', required=True)
    image_day2 = forms.ImageField(label='Day 2 Image', required=True)
    image_day3 = forms.ImageField(label='Day 3 Image', required=True)
    image_day4 = forms.ImageField(label='Day 4 Image', required=True)
    image_day5 = forms.ImageField(label='Day 5 Image', required=True)
    image_day6 = forms.ImageField(label='Day 6 Image', required=True)
