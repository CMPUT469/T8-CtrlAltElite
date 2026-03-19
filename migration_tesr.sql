-- migration_wos.sql
-- Run once against your Supabase project after pulling the refactored code.
-- Replaces all old metric columns with the single WOS metric.

-- Add new columns
ALTER TABLE test_runs
    ADD COLUMN IF NOT EXISTS wos      float,
    ADD COLUMN IF NOT EXISTS wos_l1   float,
    ADD COLUMN IF NOT EXISTS wos_l2   float,
    ADD COLUMN IF NOT EXISTS wos_l3   float;

-- Drop all old metric columns
ALTER TABLE test_runs
    DROP COLUMN IF EXISTS f1_score,
    DROP COLUMN IF EXISTS precision,
    DROP COLUMN IF EXISTS recall,
    DROP COLUMN IF EXISTS outcome_accuracy,
    DROP COLUMN IF EXISTS tesr_overall,
    DROP COLUMN IF EXISTS tesr_l1,
    DROP COLUMN IF EXISTS tesr_l2,
    DROP COLUMN IF EXISTS tesr_l3,
    DROP COLUMN IF EXISTS tsr_function_selection,
    DROP COLUMN IF EXISTS tsr_parameter_accuracy,
    DROP COLUMN IF EXISTS tsr_result_accuracy,
    DROP COLUMN IF EXISTS function_selection_rate,
    DROP COLUMN IF EXISTS parameter_accuracy,
    DROP COLUMN IF EXISTS correct_function,
    DROP COLUMN IF EXISTS correct_params,
    DROP COLUMN IF EXISTS correct_outcome;

-- Rename total_tests → total_tasks for consistency
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='test_runs' AND column_name='total_tests'
    ) THEN
        ALTER TABLE test_runs RENAME COLUMN total_tests TO total_tasks;
    END IF;
END $$;

-- Add new per-task columns to test_details
ALTER TABLE test_details
    ADD COLUMN IF NOT EXISTS level         text,
    ADD COLUMN IF NOT EXISTS optimal_steps int,
    ADD COLUMN IF NOT EXISTS actual_steps  int,
    ADD COLUMN IF NOT EXISTS call_source   text;

-- Drop old per-task columns no longer needed
ALTER TABLE test_details
    DROP COLUMN IF EXISTS correct_function,
    DROP COLUMN IF EXISTS correct_params,
    DROP COLUMN IF EXISTS expected_result;
