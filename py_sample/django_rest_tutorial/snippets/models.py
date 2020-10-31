from django.db import models
from pygments.lexers import get_all_lexers
from pygments.styles import get_all_styles

LEXERS = list(filter(lambda x: x[1], get_all_lexers()))
LANGUAGE_CHOICES = sorted(list(map(lambda x: (x[1][0], x[0]), LEXERS)))
STYLE_CHOICES = sorted(list(map(lambda x: (x, x), get_all_styles())))


class Snippet(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField()
    linenos = models.BooleanField(default=False)
    language = models.CharField(choices=LANGUAGE_CHOICES, default='python', max_length=100)
    style = models.CharField(choices=STYLE_CHOICES, default='friendly', max_length=100)

    class Meta:
        ordering = ['created']
