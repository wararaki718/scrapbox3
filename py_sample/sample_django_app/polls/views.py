from django.http import HttpResponse, HttpRequest, Http404
from django.shortcuts import render

from .models import Question

def index(request: HttpRequest) -> HttpResponse:
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    context = {
        'latest_question_list': latest_question_list
    }
    return render(request, 'polls/index.html', context)

def detail(request: HttpRequest, question_id: int) -> HttpResponse:
    try:
        question = Question.objects.get(pk=question_id)
    except Question.DoesNotExist:
        raise Http404('Question does not exists')
    return render(request, 'polls/detail.html', { 'question': question })

def results(request: HttpRequest, question_id: int) -> HttpResponse:
    response = "You're looking at question %s"
    return HttpResponse(response % question_id)

def vote(request: HttpRequest, question_id: int) -> HttpResponse:
    return HttpResponse(f"You're voting on question {question_id}")
