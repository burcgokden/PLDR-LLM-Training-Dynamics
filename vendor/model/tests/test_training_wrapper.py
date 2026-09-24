from pathlib import Path
from scripts.check_training_wrapper import parser_contract


def test_training_verifier_argument_vector(tmp_path):
    wrapper=Path(__file__).resolve().parents[1]/'scripts/workspace-wrappers/verify_training_completed.sh'
    passed=parser_contract(wrapper)
    assert passed['status']==0
    assert passed['parsed']['output']==passed['expected_output']
    assert parser_contract(wrapper,[])['status']==2
    for name,replacement in [('dropped',''),('collapsed','"$*"')]:
        mutation=tmp_path/(name+'.sh')
        mutation.write_text(wrapper.read_text().replace('"$@"',replacement))
        assert parser_contract(mutation)['status']==2
