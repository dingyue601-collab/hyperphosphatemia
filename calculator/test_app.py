"""Run with the same Python environment as app.py; no server is required."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inference import clinical_input, TwoPredictorModels


def main():
    for age, cr in [(None, 180), (65, None), (float('nan'), 180), (17, 180), (65, 0),
                    (True, 180), (65, np.complex128(180+2j)), (np.complex128(65+1j), 180)]:
        try:
            clinical_input(age, cr)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('Invalid clinical input was accepted')
    unloaded = TwoPredictorModels.__new__(TwoPredictorModels)
    for values in ((True, 180), (17, 180), (65, np.complex128(180+2j)), (65, float('inf'))):
        frame = pd.DataFrame([values], columns=['age', 'creatinine'])
        try:
            unloaded.predict_mi(frame, 1)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid native inference input was accepted')
    a = clinical_input(65, 180)
    b = clinical_input(65, 180/88.4, 'mg/dL')
    assert np.max(np.abs(a.to_numpy()-b.to_numpy())) < 1e-12
    at = AppTest.from_file(str(HERE/'app.py'), default_timeout=90).run()
    assert not at.exception and at.title[0].value == 'Type 2 diabetic kidney disease hyperphosphatemia alert'
    assert at.number_input[0].value == 65.0 and at.number_input[1].value == 180.0
    assert len(at.metric) == 1 and not at.get('expander') and not at.get('dataframe')
    assert any(x.value == 'Model information' for x in at.subheader)
    assert not any('Logistic' in x.value for x in at.markdown)
    assert any(x.value == 'Screening threshold 12.92%' for x in at.caption)
    assert any(x.value == '**Hyperphosphatemia alert**' for x in at.markdown)
    probabilities = [m.value for m in at.metric]
    assert at.get('segmented_control')[1].proto.required
    at.get('segmented_control')[1].set_value(None).run()
    assert not at.exception and at.session_state['unit'] == 'umol/L'
    assert at.number_input[1].value == 180.0
    at.get('segmented_control')[1].set_value('mg/dL').run()
    assert not at.exception
    assert np.isclose(at.number_input[1].value * 88.4, 180.0)
    at.get('segmented_control')[1].set_value(None).run()
    assert not at.exception and at.session_state['unit'] == 'mg/dL'
    assert np.isclose(at.number_input[1].value * 88.4, 180.0)
    at.button(key='calculate').click().run()
    assert not at.exception and [m.value for m in at.metric] == probabilities
    at.get('segmented_control')[1].set_value('umol/L').run()
    assert not at.exception and np.isclose(at.number_input[1].value, 180.0)
    at.button(key='calculate').click().run()
    assert not at.exception and [m.value for m in at.metric] == probabilities
    at.get('segmented_control')[0].set_value('中文').run()
    assert not at.exception and at.title[0].value == '2型糖尿病肾病高磷血症风险警示'
    assert any(x.value == '筛查阈值 12.92%' for x in at.caption)
    assert any(x.value == '\u6a21\u578b\u4fe1\u606f' for x in at.subheader)
    assert any(x.value == '**高磷血症风险警示**' for x in at.markdown)
    at.button(key='clear').click().run()
    assert not at.exception and not at.metric
    at.button(key='calculate').click().run()
    assert not at.exception and len(at.error) == 1
    at.button(key='example').click().run()
    assert not at.exception and [m.value for m in at.metric] == probabilities
    print('APPTEST_DEFAULT_ENGLISH_EXAMPLE_TABM_ONLY_UNIT_BILINGUAL_RESET_PASS', probabilities)


if __name__ == '__main__':
    main()
