import os
import json
from collections import defaultdict, Counter


# NOTE: only consider the domains and intents in the "Domains" and "WizardCapabilities" fields of each dialog
all_domains = ['weather', 'restaurant', 'bank', 'hotel', 'party', 'ride', 'trivia', 'plane', 'trip', 'doctor', 'apartment', 'spaceship', 'meeting']
all_intents = ["weather", "restaurant_book", "trivia", "bank_fraud_report", "trip_directions", "party_plan", "bank_balance", "ride_book", "plane_book", "apartment_schedule", "hotel_book", "hotel_search", "restaurant_search", "doctor_schedule", "party_rsvp", "plane_search", "ride_status", "hotel_service_request", "doctor_followup", "meeting_schedule", "apartment_search", "ride_change", "spaceship_access_codes", "spaceship_life_support"]

def read_ontology(dataset_dir):
    # TODO: manually process ontology files in 'knowledgebase/apis' folder by selecting "example values" or "possible values" for each input slot
    api_path = "apis/apis"
    api_with_wrong_required_slots = ["bank_fraud_report", "weather"]
    schema = {}
    intent_schema = {}
    domain2slots = defaultdict(set)
    for fn in os.listdir(os.path.join(dataset_dir, api_path)):
        api_name = fn.replace(".json", "")
        domain = api_name.split("_")[0]
        with open(os.path.join(dataset_dir, api_path, fn), 'r', encoding="utf-8") as f:
            try:
                ontology = json.load(f)
            except Exception as e:
                print(f"[{api_name}]: {e}")
                exit()
        if domain not in schema:
            schema[domain] = {
                "intents": []
            }
        if domain not in intent_schema:
            intent_schema[domain] = {}
        schema[domain]["intents"].append(api_name)
        intent_schema[domain][api_name] = {}
        input_slots = set([slot["Name"] for slot in ontology["input"]])
        result_slots = set([slot["Name"] for slot in ontology["output"]])
        required_slots = set(ontology["required"])
        if required_slots and api_name not in api_with_wrong_required_slots:
            assert required_slots.issubset(input_slots), f"{api_name}: {required_slots - input_slots}"
            print(f"{api_name}: input_slots ({len(input_slots)}) - required_slots ({len(required_slots)}) = {input_slots - required_slots}")
            intent_schema[domain][api_name]["required_slots"] = list(required_slots)
            intent_schema[domain][api_name]["optional_slots"] = list(input_slots - required_slots)
        else:
            intent_schema[domain][api_name]["informable slots"] = list(input_slots)
            print(f"{api_name}: no required slots.")
        intent_schema[domain][api_name]["result_slots"] = list(result_slots)
    return schema, intent_schema


def update_domain_and_intent(task, prev_turn, cur_turn, all_domains, all_intents, domain2active_intents):
    domain = task.split("_")[0]
    if domain in all_domains:
        cur_turn["turn_domain"] = [domain]
        if task in all_intents:
            cur_turn["turn_intent"] = {domain: [task]}
        else:
            if len(domain2active_intents[domain]) == 1:
                cur_turn["turn_intent"] = {domain: list(domain2active_intents[domain])}
            elif domain not in cur_turn["turn_intent"]:
                if prev_turn is not None and domain in prev_turn["turn_intent"]:
                    cur_turn["turn_intent"] = {domain: prev_turn["turn_intent"][domain]}
                else:
                    cur_turn["turn_intent"] = {domain: []}
    elif task == "" and cur_turn["turn_intent"] == {}:
        if prev_turn is not None:
            cur_turn["turn_domain"] = prev_turn["turn_domain"]
            cur_turn["turn_intent"] = prev_turn["turn_intent"]
