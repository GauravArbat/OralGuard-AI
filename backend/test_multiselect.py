import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from clinical.questionnaire import generate_questionnaire
from clinical.differential import generate_differential_diagnosis
from ai.clinical_fusion import clinical_fusion
from clinical.decision_engine import clinical_engine
import numpy as np

def test_multiselect_workflow():
    print("=" * 60)
    print("TESTING MULTI-SELECT CLINICAL QUESTIONNAIRE")
    print("=" * 60)

    questions = generate_questionnaire()
    print(f"Total questions generated: {len(questions)}")

    # 1. Verify the 4 questions are multiselect
    target_ids = ['location', 'habits', 'medical_conditions', 'medications']
    found = {q['id']: q for q in questions if q['id'] in target_ids}
    
    for qid in target_ids:
        q = found.get(qid)
        assert q is not None, f"Missing question {qid}"
        q_type = q.get("question_type")
        opts = q.get("options", [])
        print(f"[OK] [{qid}] Type: {q_type} | Options count: {len(opts)}")
        print(f"   Question: {q.get('question')}")
        print(f"   Rationale: {q.get('clinical_rationale')}")
        assert q_type == "multiselect", f"Expected multiselect for {qid}, got {q_type}"
        for opt in opts[:3]:
            print(f"     - {opt['value']}: {opt['label']}")
        print("     ...")

    # 2. Test multi-select answers evaluation in Differential Diagnosis
    print("\n" + "=" * 60)
    print("TESTING DIFFERENTIAL DIAGNOSIS WITH MULTI-SELECT ANSWERS")
    print("=" * 60)
    
    answers = {
        'location': ['labial_mucosa', 'buccal_mucosa'],
        'habits': ['smoking', 'alcohol'],
        'medical_conditions': ['anemia', 'celiac'],
        'medications': ['nsaids'],
        'onset_duration_days': 8,
        'recurrent': True,
        'pain_present': True,
        'ulcer_count': '1',
        'ulcer_size_mm': '5_to_10'
    }

    cls_res = {
        'primary_class': 'aphthous_ulcer',
        'confidence': 0.88,
        'probabilities': [
            {'label': 'aphthous_ulcer', 'probability': 0.85, 'display_name': 'Aphthous Ulcer'},
            {'label': 'oscc', 'probability': 0.08, 'display_name': 'OSCC'},
            {'label': 'other', 'probability': 0.07, 'display_name': 'Other'}
        ]
    }
    feat_res = {'border_type': 'regular', 'ulceration': True}

    diffs = generate_differential_diagnosis(cls_res, feat_res, answers)
    print(f"Top Differential: {diffs[0]['display_name']} ({diffs[0]['probability']*100:.1f}%)")
    print("Supporting features for top condition:")
    for sup in diffs[0]['supporting_features']:
        print(f"  + {sup}")

    # 3. Test Clinical Fusion with Multi-select
    print("\n" + "=" * 60)
    print("TESTING CLINICAL FUSION RISK SCORING")
    print("=" * 60)
    
    fusion = clinical_fusion.fuse(
        image_features=np.zeros(2048),
        clinical_features=feat_res,
        questionnaire_responses=answers,
        classification_result=cls_res
    )
    print(f"Fusion Risk Score: {fusion['risk_score']} / 100 ({fusion['risk_level'].upper()})")
    print("Contributing factors:")
    for factor in fusion['contributing_factors']:
        print(f"  * {factor}")

    # 4. Test Clinical Recommendations
    print("\n" + "=" * 60)
    print("TESTING CLINICAL RECOMMENDATIONS")
    print("=" * 60)
    recs = clinical_engine.generate_recommendations(
        risk_level=fusion['risk_level'],
        risk_score=fusion['risk_score'],
        primary_diagnosis=cls_res['primary_class'],
        differentials=diffs,
        questionnaire_responses=answers,
        clinical_features=feat_res
    )
    for i, r in enumerate(recs, 1):
        print(f" {i}. {r}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_multiselect_workflow()
