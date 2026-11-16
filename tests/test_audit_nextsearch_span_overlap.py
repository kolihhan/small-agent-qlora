from scripts.audit_nextsearch_vs_gaia_bird import norm

def test_normalized_reference_is_found_inside_training_payload():
    reference = 'SELECT COUNT(*) FROM Customers WHERE city = "Paris"'
    payload = 'The diagnostic SQL was: ' + reference + '; and here are results.'
    assert norm(reference) in norm(payload)

def test_reference_absent_from_unrelated_payload():
    reference = "Which city hosted the robotics conference in the year 2019?"
    payload = "A different task asks for the number of roads in Italy."
    assert norm(reference) not in norm(payload)
