"""
RAG Ground Truth Dataset - 20 Questions (10 per manual)
Based on actual text data from car manuals
"""

import json
from typing import List, Dict


# =============================================================================
# TATA TIAGO GROUND TRUTH - 10 QUESTIONS
# =============================================================================

TATA_TIAGO_GROUND_TRUTH = [
    {
        "id": "TIAGO_Q1",
        "query": "What engine oil specification is recommended for Tata Tiago?",
        "car_model": "Tata Tiago",
        "expected_answer": "The recommended engine oil specification for Tata Tiago is 5W30 ACEA A5/B5 conforming to TATA SS6579. Recommended brands include CASTROL Magnatec Professional T 5W30 and PETRONAS TATA MOTORS Genuine Oil Engine Oil Synth 5W30. The capacity is 3.5 litres for petrol variant and 4 litres for diesel variant.",
        "key_facts": [
            "5W30 ACEA A5/B5",
            "TATA SS6579",
            "CASTROL Magnatec Professional T 5W30",
            "PETRONAS TATA MOTORS Genuine Oil",
            "3.5 litres for petrol",
            "4 litres for diesel"
        ],
        "relevant_chunks": [
            "Engine Oil 5W30 ACEA A5/B5 TATA SS6579 CASTROL - Magnatec Professional T 5W30 PETRONAS TATA MOTORS Genuine Oil – Engine Oil Synth 5W30 3.5 Litres (Petrol) 4 Litres (Diesel)",
            "Use following genuine fluids, coolants and lubricants recommended for optimum performance of your vehicle."
        ],
        "page_references": ["Page 127 - Lubricant Specification"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    },
    
    {
        "id": "TIAGO_Q2",
        "query": "What are the engine specifications for Tata Tiago diesel variant?",
        "car_model": "Tata Tiago",
        "expected_answer": "The Tata Tiago diesel variant has a 1.05L REVOTORQ engine with a capacity of 1047 cc. It produces a maximum engine output of 70 PS at 4000 +/-50 rpm and maximum torque of 140 Nm at 1800-3000 rpm.",
        "key_facts": [
            "1.05L REVOTORQ",
            "1047 cc",
            "70 PS @ 4000 +/-50 rpm",
            "140 Nm @ 1800-3000 rpm"
        ],
        "relevant_chunks": [
            "Model/type 1.05L REVOTORQ Capacity 1047 cc Max. Engine output 70 PS @ 4000 +/-50 rpm Max. Torque 140 Nm @ 1800 - 3000 rpm",
            "Parameter Diesel Petrol Engine Model/type 1.05L REVOTORQ 1.2L REVOTRON"
        ],
        "page_references": ["Page 128 - Technical Specifications"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    },
    
    {
        "id": "TIAGO_Q3",
        "query": "What is the warranty period for Tata Tiago?",
        "car_model": "Tata Tiago",
        "expected_answer": "The warranty for Tata Tiago is for a period of 24 months from the date of sale or a mileage of 75,000 km, whichever occurs earlier.",
        "key_facts": [
            "24 months",
            "75,000 km",
            "whichever occurs earlier",
            "from date of sale"
        ],
        "relevant_chunks": [
            "This warranty shall be for a period of 24 months from the date of sale of the car or a mileage of 75,000 Kms whichever occurs earlier.",
            "We WARRANT each TATA TIAGO vehicle and parts thereof manufactured by us to be free from defect in material and workmanship subject to the following terms and conditions"
        ],
        "page_references": ["Page 147 - Warranty Terms and Conditions"],
        "question_type": "factual",
        "difficulty": "easy"
    },
    
    {
        "id": "TIAGO_Q4",
        "query": "What type of fuel should be used for Tata Tiago petrol variant?",
        "car_model": "Tata Tiago",
        "expected_answer": "For Tata Tiago petrol variant, unleaded regular grade petrol conforming to IS 2796:2008 and BS IV specification with RON (Research Octane Number) not less than 91 is recommended. Always use petrol of correct specification in a vehicle fitted with catalytic converter, as even a single fill of leaded petrol will seriously damage the catalytic converter.",
        "key_facts": [
            "Unleaded regular grade petrol",
            "IS 2796:2008",
            "BS IV specification",
            "RON not less than 91",
            "Do not use leaded petrol"
        ],
        "relevant_chunks": [
            "Unleaded regular grade petrol conforming to IS 2796:2008 and BS IV specification and RON not less than 91 is recommended.",
            "Always use petrol of a correct specification in a vehicle fitted with catalytic converter. Even single fill of leaded petrol will seriously damage the catalytic converter."
        ],
        "page_references": ["Page 126 - Fuel (Petrol)"],
        "question_type": "technical_specification",
        "difficulty": "medium"
    },
    
    {
        "id": "TIAGO_Q5",
        "query": "What is the brake fluid specification for Tata Tiago?",
        "car_model": "Tata Tiago",
        "expected_answer": "The brake fluid specification for Tata Tiago is IS 8654 Type II DOT 4 conforming to TATA SS7711. Recommended brands include PETRONAS Tutela TOP 45 DOT 4, CASTROL UBF DOT 4, and SUNSTAR CCI Golden Cruiser Brake Fluid DOT 4. The capacity is 0.5 litres.",
        "key_facts": [
            "IS 8654 Type II DOT 4",
            "TATA SS7711",
            "0.5 litres",
            "PETRONAS Tutela TOP 45 DOT 4",
            "CASTROL UBF DOT 4"
        ],
        "relevant_chunks": [
            "Brake Fluid IS 8654 Type II DOT 4 TATA SS7711 PETRONAS - Tutela - TOP 45 DOT 4 CASTROL - UBF DOT 4 SUNSTAR CCI - Golden Cruiser Brake Fluid DOT 4 0.5 Litres"
        ],
        "page_references": ["Page 127 - Lubricant Specification"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    },
    
    {
        "id": "TIAGO_Q6",
        "query": "How do I turn on the indicators in Tata Tiago?",
        "car_model": "Tata Tiago",
        "expected_answer": "To turn on the indicators in Tata Tiago: For left turn signal, move the right-hand stalk lever fully upward. For right turn signal, move the lever fully downward. When the turn is completed, the signal will cancel automatically and the lever will return to its normal position. For lane change signal, move the lever slightly up or down without full latching, and the turn signal will flash 3 times automatically.",
        "key_facts": [
            "Left turn: move lever fully upward",
            "Right turn: move lever fully downward",
            "Auto-cancels when turn completed",
            "Lane change: slight movement, 3 flashes"
        ],
        "relevant_chunks": [
            "1. Left Turn signal - Move the lever fully upward. 2. Right Turn signal - Move the lever fully downward.",
            "When the turn is completed, the signal will cancel and the lever will return to its normal position.",
            "To signal a lane change, move the lever slightly up or down respective to the point where the turn signal light begins to flash, but the lever does not latch. The turn signal will flash 3 times automatically."
        ],
        "page_references": ["Page 47 - Combi-Switch (RH Stalk)"],
        "question_type": "procedural",
        "difficulty": "medium"
    },
    
    {
        "id": "TIAGO_Q7",
        "query": "What is the fuel tank capacity of Tata Tiago?",
        "car_model": "Tata Tiago",
        "expected_answer": "The fuel tank capacity of Tata Tiago is 35 litres for both petrol and diesel variants.",
        "key_facts": [
            "35 litres",
            "Same for petrol and diesel"
        ],
        "relevant_chunks": [
            "Fuel tank Capacity 35 liters 35 liters"
        ],
        "page_references": ["Page 129 - Technical Information"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    },
    
    {
        "id": "TIAGO_Q8",
        "query": "What is the wheelbase of Tata Tiago?",
        "car_model": "Tata Tiago",
        "expected_answer": "The wheelbase of Tata Tiago is 2400 mm for both petrol and diesel variants.",
        "key_facts": [
            "2400 mm",
            "Same for both variants"
        ],
        "relevant_chunks": [
            "Wheel base 2400 2400",
            "Main chassis dimension (in mm) Wheel base 2400 2400"
        ],
        "page_references": ["Page 129-130 - Main Chassis Dimensions"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    },
    
    {
        "id": "TIAGO_Q9",
        "query": "What type of clutch does Tata Tiago have?",
        "car_model": "Tata Tiago",
        "expected_answer": "Tata Tiago has a single plate dry friction diaphragm type clutch with an outside diameter of 200 mm.",
        "key_facts": [
            "Single plate dry friction diaphragm type",
            "200 mm diameter"
        ],
        "relevant_chunks": [
            "Clutch Type Single plate dry friction diaphragm type Outside diameter of clutch 200 mm"
        ],
        "page_references": ["Page 128 - Technical Specifications"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    },
    
    {
        "id": "TIAGO_Q10",
        "query": "What is the ground clearance of Tata Tiago?",
        "car_model": "Tata Tiago",
        "expected_answer": "The ground clearance of Tata Tiago is 170 mm for both petrol and diesel variants.",
        "key_facts": [
            "170 mm",
            "Same for both variants"
        ],
        "relevant_chunks": [
            "Ground clearance 170 170",
            "Main chassis dimension (in mm) Ground clearance 170 170"
        ],
        "page_references": ["Page 129-130 - Main Chassis Dimensions"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    }
]


# =============================================================================
# MG ASTOR GROUND TRUTH - 10 QUESTIONS
# =============================================================================

MG_ASTOR_GROUND_TRUTH = [
    {
        "id": "ASTOR_Q1",
        "query": "How do I check the engine oil level in MG Astor?",
        "car_model": "MG Astor",
        "expected_answer": "To check the engine oil level in MG Astor: Check the oil level weekly, ideally with the engine cold and car on level ground. If the engine is warm, wait at least five minutes after switching off before checking. Withdraw the dipstick and wipe off the oil. Slowly insert the oil dipstick and pull it out to check the level - it should not be lower than the MIN mark. If needed, screw off the oil filler cap and refill to maintain level between MAX and MIN marks. Wait 5 minutes and recheck, adding oil if necessary. Do not overfill. Ensure dipstick is inserted and oil filler cap is fully secured.",
        "key_facts": [
            "Check weekly",
            "Engine cold, level ground",
            "Wait 5 minutes if engine warm",
            "Wipe dipstick clean",
            "Level between MIN and MAX",
            "Do not overfill",
            "Secure cap after filling"
        ],
        "relevant_chunks": [
            "Check the oil level weekly and top up with oil when necessary. Ideally, the oil level should be checked with the engine cold and the car resting on level ground. However, if the engine is running and already getting warm, wait for at least five minutes after switching off the START/STOP Switch before checking the level.",
            "Withdraw the dipstick and wipe off the oil on it. Slowly insert the oil dipstick and pull it out again to check the oil level; the oil level shall not be lower than the MIN mark on the oil dipstick.",
            "Screw off the oil filler cap and refill the oil to maintain the oil level between the MAX mark and MIN mark on the oil dipstick. Wait for 5 minutes and then recheck the oil level, add an appropriate amount of oil if necessary – DO NOT OVERFILL!"
        ],
        "page_references": ["Page 206-207 - Maintenance, Checking Engine Oil"],
        "question_type": "procedural",
        "difficulty": "hard"
    },
    
    {
        "id": "ASTOR_Q2",
        "query": "What engine oil specification should I use for MG Astor?",
        "car_model": "MG Astor",
        "expected_answer": "For MG Astor, use the engine oil recommended and certified by the manufacturer. Refer to 'Recommended Fluids and Capacities' in the Technical Data section for specific specifications. Do not use oil additives not applicable to the car as they may damage the engine. Only use oil additives certified by the manufacturer - consult your local Authorised Dealer for details.",
        "key_facts": [
            "Use manufacturer recommended oil",
            "Refer to Technical Data section",
            "Check Recommended Fluids and Capacities",
            "Do not use unapproved additives",
            "Consult dealer for certified additives"
        ],
        "relevant_chunks": [
            "Use the engine oil recommended and certified by the manufacturer. Refer to Recommended Fluids and Capacities in Technical Data section.",
            "Do not use the oil additives not applicable to the car, or else the engine may be damaged. You are recommended to use the oil additives certified by the manufacturer, please consult your local Authorised Dealer for details."
        ],
        "page_references": ["Page 207 - Engine Oil Specification"],
        "question_type": "technical_specification",
        "difficulty": "medium"
    },
    
    {
        "id": "ASTOR_Q3",
        "query": "How do I turn on the indicators in MG Astor?",
        "car_model": "MG Astor",
        "expected_answer": "To turn on indicators in MG Astor: Move the lever down to indicate a LEFT turn. Move the lever up to indicate a RIGHT turn. The corresponding green indicator lamp in the instrument pack will flash when the turning signal lamps are working. Rotating the steering wheel will cancel the indicator operation automatically (small movements may not operate self-cancelling). To indicate a lane change, move the lever briefly and release - the indicators will flash three times and then cancel.",
        "key_facts": [
            "Lever down for LEFT turn",
            "Lever up for RIGHT turn",
            "Green indicator lamp flashes",
            "Auto-cancels with steering wheel rotation",
            "Lane change: brief movement, 3 flashes"
        ],
        "relevant_chunks": [
            "Move the lever down to indicate a LEFT turn (1). Move the lever up to indicate a RIGHT turn (2). The corresponding GREEN indicator lamp in the instrument pack will flash when the turning signal lamps are working.",
            "Rotating the steering wheel will cancel the indicator operation (small movements of the steering wheel may not operate the self cancelling). To indicate a lane change, move the lever briefly and release, the indicators will flash three times and then cancel."
        ],
        "page_references": ["Page 37-38 - Direction Indicators"],
        "question_type": "procedural",
        "difficulty": "easy"
    },
    
    {
        "id": "ASTOR_Q4",
        "query": "What is the coolant specification for MG Astor?",
        "car_model": "MG Astor",
        "expected_answer": "For MG Astor, use the coolant which is recommended and certified by the manufacturer. Refer to 'Recommended Fluids and Capacities' in the Technical Data section for specific specifications. Do not add corrosion inhibitors or other additives to the cooling system as they may severely disrupt the efficiency of the system and cause parts damage. Coolant is poisonous and can be fatal if swallowed - keep containers sealed and away from children.",
        "key_facts": [
            "Use manufacturer recommended coolant",
            "Refer to Technical Data section",
            "No corrosion inhibitors or additives",
            "Can disrupt system efficiency",
            "Coolant is poisonous",
            "Keep away from children"
        ],
        "relevant_chunks": [
            "Please use the coolant which is recommended and certified. Please refer to Recommended Fluids and Capacities in the Technical Data section.",
            "The addition of corrosion inhibitors or other additives to the cooling system of this car may severely disrupt the efficiency of the system and cause parts damage.",
            "Coolant is poisonous and can be fatal if swallowed - keep coolant containers sealed and out of the reach of children."
        ],
        "page_references": ["Page 207-208 - Coolant Specification"],
        "question_type": "technical_specification",
        "difficulty": "medium"
    },
    
    {
        "id": "ASTOR_Q5",
        "query": "What is the recommended engine oil capacity for MG Astor 220 TURBO variant?",
        "car_model": "MG Astor",
        "expected_answer": "The recommended engine oil capacity for MG Astor 220 TURBO (6AT) variant is 4.6 litres for after-sales replacement. The specification is C2 5W-30.",
        "key_facts": [
            "4.6 litres",
            "220 TURBO (6AT) variant",
            "C2 5W-30 specification",
            "After-sales replacement"
        ],
        "relevant_chunks": [
            "Engine oil (after-sales replacement), L C2 5W-30 4.1 4.6",
            "Name Grade Capacity VTi - TECH (5MT) VTi - TECH (CVT) 220 TURBO (6AT) Engine oil (after-sales replacement), L C2 5W-30 - - 4.6"
        ],
        "page_references": ["Page 228 - Recommended Fluids and Capacities"],
        "question_type": "technical_specification",
        "difficulty": "medium"
    },
    
    {
        "id": "ASTOR_Q6",
        "query": "What is the engine capacity and power output of MG Astor 1.5L engine?",
        "car_model": "MG Astor",
        "expected_answer": "The MG Astor 1.5L engine (VTi-TECH) has a capacity of 1.498 litres with bore x stroke of 75mm x 84.8mm. It has a compression ratio of 11.5:1 and uses INDIA 91 unleaded gasoline and above as fuel type.",
        "key_facts": [
            "1.498 litres capacity",
            "Bore x stroke: 75mm x 84.8mm",
            "Compression ratio 11.5:1",
            "INDIA 91 unleaded gasoline"
        ],
        "relevant_chunks": [
            "Vehicle Parameter VTi - TECH (5MT) / CVT 220 TURBO (6AT) Bore × Stroke, mm × mm 75×84.8 80×89.4 Capacity, Litres 1.498 1.349 Compression Ratio 11.5:1 10:1 Fuel Type INDIA 91 unleaded gasoline and above"
        ],
        "page_references": ["Page 227 - Major Parameters of Engine"],
        "question_type": "technical_specification",
        "difficulty": "medium"
    },
    
    {
        "id": "ASTOR_Q7",
        "query": "What is the tyre pressure specification for MG Astor?",
        "car_model": "MG Astor",
        "expected_answer": "The tyre pressure specification for MG Astor in cold condition is 230kPa/2.3bar/32psi for both front and rear wheels when unladen. It is recommended that the pressure of spare tyre should be consistent with that of main tyre.",
        "key_facts": [
            "230kPa or 2.3bar or 32psi",
            "Same for front and rear wheels",
            "Cold condition",
            "Unladen",
            "Spare tyre same as main tyre"
        ],
        "relevant_chunks": [
            "Tyre Pressure (Cold) Wheels Unladen Front Wheels 230kPa/2.3bar/32psi Rear Wheels 230kPa/2.3bar/32psi",
            "It is recommended that the pressure of spare tyre should be consistent with that of main tyre."
        ],
        "page_references": ["Page 229 - Tyre Pressure"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    },
    
    {
        "id": "ASTOR_Q8",
        "query": "What is the warranty period for MG Astor battery?",
        "car_model": "MG Astor",
        "expected_answer": "The warranty coverage for 12V/48V Battery in MG Astor is valid for 1 year starting from the Delivery Date shown in the Owner's Manual issued to the Customer. The warranty is provided by the battery manufacturer as per their terms and conditions.",
        "key_facts": [
            "1 year warranty",
            "From delivery date",
            "12V/48V battery",
            "Provided by battery manufacturer"
        ],
        "relevant_chunks": [
            "Battery: The warranty coverage for 12V / 48V Battery is valid for 1 year starting from the Delivery Date shown in the Owner's Manual issued to the Customer and shall be provided by the battery manufacturer as per their terms and conditions."
        ],
        "page_references": ["Page 236 - Parts not covered under warranty"],
        "question_type": "factual",
        "difficulty": "easy"
    },
    
    {
        "id": "ASTOR_Q9",
        "query": "What is the ground clearance of MG Astor 220 TURBO variant?",
        "car_model": "MG Astor",
        "expected_answer": "The ground clearance of MG Astor 220 TURBO (6AT) variant is 141 mm in laden condition.",
        "key_facts": [
            "141 mm",
            "220 TURBO (6AT) variant",
            "Laden condition"
        ],
        "relevant_chunks": [
            "Ground Clearance (mm - Laden condition) 151 152 141",
            "Item, Units VTi - TECH (5MT) VTi - TECH (CVT) 220 TURBO (6AT) Ground Clearance (mm - Laden condition) 151 152 141"
        ],
        "page_references": ["Page 226 - Weights"],
        "question_type": "technical_specification",
        "difficulty": "easy"
    },
    
    {
        "id": "ASTOR_Q10",
        "query": "How should I check the coolant level in MG Astor?",
        "car_model": "MG Astor",
        "expected_answer": "The cooling system should be checked weekly when the cooling system is cold and with the car resting on level ground. Do not remove the coolant pressure cap when the cooling system is hot as escaping steam or hot coolant could cause serious injury. If the coolant level is below the MIN mark, open the coolant expansion tank cap and top up coolant. The coolant level should not be higher than the MAX mark. Prevent coolant from coming into contact with the vehicle body when topping up as coolant will damage paint.",
        "key_facts": [
            "Check weekly when cold",
            "Level ground required",
            "Do not remove cap when hot",
            "Top up if below MIN mark",
            "Do not exceed MAX mark",
            "Prevent coolant contact with body"
        ],
        "relevant_chunks": [
            "DO NOT remove the coolant pressure cap when the cooling system is hot - escaping steam or hot coolant could cause serious injury.",
            "The cooling system should be checked weekly when the cooling system is cold and with the car resting on level ground. If the coolant level is below the MIN mark, open the coolant expansion tank cap and top up coolant. The coolant level should not be higher than the MAX mark.",
            "Prevent coolant from coming into contact with the vehicle body when topping up. Coolant will damage paint."
        ],
        "page_references": ["Page 207-208 - Coolant Check and Top Up"],
        "question_type": "procedural",
        "difficulty": "medium"
    }
]


# =============================================================================
# COMBINED DATASET
# =============================================================================

GROUND_TRUTH_DATASET = {
    "metadata": {
        "total_questions": 20,
        "tata_tiago_questions": 10,
        "mg_astor_questions": 10,
        "description": "Ground truth dataset for RAG evaluation based on actual text from car manuals",
        "data_source": "Extracted from PDF manuals - text only, no images or tables",
        "question_types": {
            "technical_specification": "Questions about specifications, capacities, and technical details",
            "procedural": "Questions about how to perform tasks or operations",
            "factual": "Questions about facts, warranty, and general information"
        }
    },
    "tata_tiago": TATA_TIAGO_GROUND_TRUTH,
    "mg_astor": MG_ASTOR_GROUND_TRUTH,
    "all_questions": TATA_TIAGO_GROUND_TRUTH + MG_ASTOR_GROUND_TRUTH
}


def save_ground_truth(filename="ground_truth_dataset.json"):
    """Save ground truth dataset to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(GROUND_TRUTH_DATASET, f, indent=2, ensure_ascii=False)
    print(f"Ground truth dataset saved to {filename}")
    return filename


def print_summary():
    """Print summary of ground truth dataset"""
    print("=" * 80)
    print("GROUND TRUTH DATASET SUMMARY")
    print("=" * 80)
    print(f"\nTotal Questions: {GROUND_TRUTH_DATASET['metadata']['total_questions']}")
    print(f"  - Tata Tiago: {GROUND_TRUTH_DATASET['metadata']['tata_tiago_questions']}")
    print(f"  - MG Astor: {GROUND_TRUTH_DATASET['metadata']['mg_astor_questions']}")
    
    print("\nQuestion Distribution by Type:")
    types = {}
    for q in GROUND_TRUTH_DATASET['all_questions']:
        qtype = q['question_type']
        types[qtype] = types.get(qtype, 0) + 1
    
    for qtype, count in types.items():
        print(f"  - {qtype}: {count}")
    
    print("\nDifficulty Distribution:")
    difficulties = {}
    for q in GROUND_TRUTH_DATASET['all_questions']:
        diff = q['difficulty']
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    for diff, count in difficulties.items():
        print(f"  - {diff}: {count}")
    
    print("\n" + "=" * 80)
    print("Sample Questions:")
    print("=" * 80)
    
    print("\n[TATA TIAGO - Sample]")
    print(f"Q: {TATA_TIAGO_GROUND_TRUTH[0]['query']}")
    print(f"Key Facts: {', '.join(TATA_TIAGO_GROUND_TRUTH[0]['key_facts'][:3])}")
    
    print("\n[MG ASTOR - Sample]")
    print(f"Q: {MG_ASTOR_GROUND_TRUTH[0]['query']}")
    print(f"Key Facts: {', '.join(MG_ASTOR_GROUND_TRUTH[0]['key_facts'][:3])}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Save ground truth to JSON
    filename = save_ground_truth()
    
    # Print summary
    print("\n")
    print_summary()
    
    print(f"\n\nDataset saved to: {filename}")
    print("You can now use this ground truth for evaluating your RAG system.")