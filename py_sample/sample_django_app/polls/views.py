from django.http import HttpResponse, HttpRequest
from django.template import loader

from .models import Question

def index(request: HttpRequest) -> HttpResponse:
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    template = loader.get_template('polls/index.html')
    context = {
        'latest_question_list': latest_question_list
    }
    return HttpResponse(template.render(context, request))

def detail(request: HttpRequest, question_id: int) -> HttpResponse:
    return HttpResponse(f"You're looking at question {question_id}")

def results(request: HttpRequest, question_id: int) -> HttpResponse:
    response = "You're looking at question %s"
    return HttpResponse(response % question_id)

def vote(request: HttpRequest, question_id: int) -> HttpResponse:
    return HttpResponse(f"You're voting on question {question_id}")
