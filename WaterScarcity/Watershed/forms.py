from django import forms

class NpyUploadForm(forms.Form):
    npyfile = forms.FileField()
