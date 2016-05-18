import sys
from wit import Wit
import pdb

# Quickstart example
# See https://wit.ai/l5t/Quickstart

if len(sys.argv) != 2:
    print("usage: python examples/quickstart.py <wit-token>")
    sys.exit(1)
access_token = sys.argv[1]

def first_entity_value(entities, entity):
    if entity not in entities:
        return None
    val = entities[entity][0]['value']
    if not val:
        return None
    return val['value'] if isinstance(val, dict) else val

def say(session_id, context, msg):
    print(msg)

def merge(session_id, context, entities, msg):
    #pdb.set_trace()
    if 'intent' in entities:    
        context['intent'] = entities['intent']
    datetime = first_entity_value(entities, 'datetime')
    if datetime:
        context['datetime'] = datetime
    userinfo = first_entity_value(entities, 'userinfo')
    if userinfo:
        context['userinfo'] = userinfo
    todo = first_entity_value(entities, 'todo')
    if todo:
        context['todo'] = todo
    
    #pdb.set_trace()
    print entities
    print context
    return context

def error(session_id, context, e):
    print(str(e))

def set_alarm(session_id, context):
    #pdb.set_trace()
    print('set-alarm('+context['datetime']+')')
    del context['intent']
    if 'yes_no' in context:
        del context['yes_no']
    context['AlarmRecentlySet'] = True
    return context

def set_todo(session_id, context):
    #pdb.set_trace()
    print('set-todo('+context['todo']+','+context['datetime']+')')
    del context['intent']
    return context
    
def play_info(session_id, context):
    #pdb.set_trace()    
    print('play-info('+context['userinfo'],','+context['datetime']+')')
    del context['intent']
    return context

actions = {
    'say': say,
    'merge': merge,
    'error': error,
    'set-alarm': set_alarm,
    'set-todo': set_todo,
    'play-info': play_info,
}

client = Wit(access_token, actions)
client.interactive()
