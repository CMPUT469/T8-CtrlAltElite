-- Supabase schema for T8-CtrlAltElite test result logging
-- Paste this into the Supabase SQL Editor and click "Run"

CREATE TABLE IF NOT EXISTS test_runs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model           text NOT NULL,
    timestamp       timestamptz NOT NULL,
    test_suite      text NOT NULL,           -- 'bfcl', 'threshold', 'incremental'
    num_distractors integer,                 -- null unless incremental/threshold run
    f1_score        float,
    precision       float,
    recall          float,
    tsr_function_selection float,
    tsr_result_accuracy    float,
    total_tests     integer,
    correct_function integer,
    correct_result  integer,
    no_tool_call    integer,
    wrong_tool      integer,
    -- Keep each failure mode queryable without unpacking raw_metrics.
    wrong_params    integer,
    raw_metrics     jsonb                    -- full metrics blob for any extra fields
);

CREATE TABLE IF NOT EXISTS test_details (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              uuid NOT NULL REFERENCES test_runs(id) ON DELETE CASCADE,
    test_id             text,
    query               text,
    expected_function   text,
    actual_function     text,
    expected_params     jsonb,
    actual_params       jsonb,
    actual_result       float,
    correct_function    boolean,
    correct_params      boolean,
    correct_result      boolean,
    error               text
);

-- Index for fast lookup by model or suite
CREATE INDEX IF NOT EXISTS idx_test_runs_model ON test_runs(model);
CREATE INDEX IF NOT EXISTS idx_test_runs_suite ON test_runs(test_suite);
CREATE INDEX IF NOT EXISTS idx_test_details_run_id ON test_details(run_id);
