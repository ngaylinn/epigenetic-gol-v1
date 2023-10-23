"""Run experiments to evolve species for various fitness goals.

This script runs evolutionary experiments specified in the experiments module.
Each experiment is broken down into multiple trials, which are mostly just
batches of work to run on the GPU. This script runs trial after trial until all
experiments are complete. It uses a breadth-first approach and calls the
visualize_results script in order to produce partial results quickly.
"""
import datetime
import signal
import subprocess

from experiments import NUM_TRIALS, experiment_list


user_requested_abort = False

def handle_sigint(sig, frame):
    """Allow the user to abort this script, gracefully or abruptly."""
    global user_requested_abort
    if user_requested_abort:
        print('Caught Ctrl+C again, terminating immediately.')
        exit(0)
    else:
        user_requested_abort = True
        print('Caught Ctrl+C, will terminate after current trial ends.')


def format_time(time_in_seconds):
    """Summarize time_in_seconds as a concise text string."""
    result = str(datetime.timedelta(seconds=int(time_in_seconds)))
    if result.startswith('0:'):
        result = result[2:]
    return result


def print_status_summary(completed_trials, elapsed_secs):
    """Prints a summary of overall progress after each trial."""
    total_trials = len(experiment_list) * NUM_TRIALS
    percent_done = 100 * completed_trials // total_trials
    print(f'Completed {completed_trials} of '
          f'{total_trials} trials ({percent_done}%) | '
          f'Elapsed: {format_time(elapsed_secs)} | '
          f'Remaining: {format_time(estimate_remaining_time())}')


def estimate_remaining_time():
    """Use historical data to estimate overalltime remaining."""
    # If there's no historical runtime data, assume a trial will take 20
    # minutes to run, which was typical in the dev environment.
    trial_duration_estimate = 20 * 60
    result = 0
    # Look at the experiments with the most completed trials first, since they
    # may have historical runtime data from this machine.
    for experiment in sorted(experiment_list, reverse=True):
        # If the current experiment has historical data, use that to estimate
        # how long it takes to run the remaining trials of this experiment.
        # This is the most accurate, since different PhenotypePrograms and
        # FitnessGoals have different performance characteristics.
        if experiment.average_trial_duration:
            # If some experiments haven't started yet but others have, then use
            # the true runtime from one experiment as an estimate for any
            # unstarted experiments.
            trial_duration_estimate = experiment.average_trial_duration
        result += trial_duration_estimate * experiment.remaining_trials
    return result


def run_experiments():
    """Execute trial after trial until all experiments are complete."""
    # Register a signal handler to deal with Ctrl+C gracefully.
    signal.signal(signal.SIGINT, handle_sigint)

    # Review the experiment list, look for work has already been done (if any),
    # and print a summary of the work left to do.
    total_experiments = 0
    completed_trials = 0
    elapsed_secs = 0
    for experiment in experiment_list:
        total_experiments += 1
        completed_trials += experiment.trial + 1
        elapsed_secs += experiment.elapsed
    print(f'Running {len(experiment_list)} experiments '
          f'of {NUM_TRIALS} trials each.')
    print_status_summary(completed_trials, elapsed_secs)

    # Keep running trials until they're all done (or the user aborts). Always
    # pick the experiment with the fewest trials to run next, so that no
    # experiment gets ahead of the others and partial results come in faster.
    while not user_requested_abort:
        # If the experiment with the fewest completed trials is done, then all
        # experiments must be done, so stop here.
        experiment = min(experiment_list)
        if experiment.has_finished():
            print('All experiments completed.')
            break

        # Run the experiment and print an updated status summary.
        print(f'Running experiment {experiment.name}, '
              f'trial {experiment.trial + 1}')
        experiment.run_trial()
        completed_trials += 1
        elapsed_secs += experiment.elapsed
        print_status_summary(completed_trials, elapsed_secs)

        # Summarize the results from this trial in a human-friendly form. This
        # is done by launching a separate script in a separate process. The
        # main reason for this is because matplotlib is prone to memory leaks
        # and isn't good for use in a long-running process like this one. In
        # theory, it also allows both scripts to run in parallel to save time
        # (though the Python GIL means this is not guaranteed).
        subprocess.Popen(['python3', 'visualize_results.py'])


if __name__ == '__main__':
    run_experiments()
