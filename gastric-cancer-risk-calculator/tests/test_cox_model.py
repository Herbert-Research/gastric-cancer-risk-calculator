def test_cox_model_performance():
    # Sample data for testing
    sample_data = {
        'time': [5, 10, 15, 20],
        'event': [1, 0, 1, 0],
        'age': [50, 60, 70, 80],
        'gender': [0, 1, 0, 1]
    }
    
    # Expected Brier score for the Cox model
    expected_brier_score = 0.25  # Replace with the actual expected value

    # Fit the Cox model and calculate the Brier score
    cox_model = CoxPHFitter()
    cox_model.fit(pd.DataFrame(sample_data), duration_col='time', event_col='event')
    brier_score = calculate_brier_score(cox_model, sample_data)

    assert brier_score == expected_brier_score, f"Expected Brier score: {expected_brier_score}, but got: {brier_score}"