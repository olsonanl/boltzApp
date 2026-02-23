#!/usr/bin/env perl

=head1 NAME

App-Boltz - BV-BRC AppService script for Boltz biomolecular structure prediction

=head1 SYNOPSIS

    App-Boltz [--preflight] params.json

=head1 DESCRIPTION

This script implements the BV-BRC AppService interface for running Boltz
biomolecular structure predictions. It handles:

- Input validation and format detection (YAML/FASTA)
- Workspace file download/upload
- Resource estimation for job scheduling
- Execution of boltz predict command
- Result collection and workspace upload

=cut

use strict;
use warnings;
use Carp::Always;  # Stack traces on errors (production debugging)
use Data::Dumper;
use File::Basename;
use File::Path qw(make_path);
use File::Slurp;
use File::Copy;
use JSON;
use Getopt::Long;
use Try::Tiny;
use POSIX qw(strftime);

# BV-BRC modules
use Bio::KBase::AppService::AppScript;

# Default log level for production
$ENV{P3_LOG_LEVEL} //= 'INFO';

# Initialize the AppScript with our callbacks
my $script = Bio::KBase::AppService::AppScript->new(\&run_boltz, \&preflight);
$script->run(\@ARGV);

=head2 preflight

Estimate resource requirements based on input parameters.

=cut

sub preflight {
    my ($app, $app_def, $raw_params, $params) = @_;

    # Default resource estimates for GPU-based structure prediction
    my $cpu = 8;
    my $memory = "64G";
    my $runtime = 7200;  # 2 hours default
    my $storage = "50G";

    # Adjust based on parameters
    my $diffusion_samples = $params->{diffusion_samples} // 1;
    my $recycling_steps = $params->{recycling_steps} // 3;

    # More samples = more time and memory
    if ($diffusion_samples > 5) {
        $runtime = 14400;  # 4 hours
        $memory = "96G";
    } elsif ($diffusion_samples > 1) {
        $runtime = 10800;  # 3 hours
        $memory = "80G";
    }

    # More recycling steps = more time
    if ($recycling_steps > 5) {
        $runtime += 3600;
    }

    # Check if affinity prediction is enabled (requires more resources)
    if ($params->{predict_affinity}) {
        $memory = "96G";
        $runtime += 1800;
    }

    return {
        cpu => $cpu,
        memory => $memory,
        runtime => $runtime,
        storage => $storage,
        policy_data => {
            gpu_count => 1,
            partition => 'gpu2',
            constraint => 'A100|H100|H200'
        }
    };
}

=head2 run_boltz

Main execution function for Boltz structure prediction.

=cut

sub run_boltz {
    my ($app, $app_def, $raw_params, $params) = @_;

    print "Starting Boltz structure prediction\n";
    print STDERR "Parameters: " . Dumper($params) . "\n" if $ENV{P3_DEBUG};

    # Create working directories
    my $work_dir = $ENV{P3_WORKDIR} // $ENV{TMPDIR} // "/tmp";
    my $input_dir = "$work_dir/input";
    my $output_dir = "$work_dir/output";

    make_path($input_dir, $output_dir);

    # Download input file from workspace
    my $input_file = $params->{input_file};
    die "Input file is required\n" unless $input_file;

    print "Downloading input file: $input_file\n";
    my $local_input = download_workspace_file($app, $input_file, $input_dir);

    # Detect input format
    my $input_format = detect_input_format($local_input, $params->{input_format});
    print "Detected input format: $input_format\n";

    # For FASTA files, rewrite headers to Boltz-compatible format
    # Boltz expects: >CHAIN_ID|ENTITY_TYPE (e.g., >A|protein)
    my $mapping_file;
    if ($input_format eq "fasta") {
        print "Rewriting FASTA identifiers for Boltz...\n";
        ($local_input, $mapping_file) = rewrite_fasta_for_boltz($local_input, $input_dir);
    } else {
        # Boltz requires specific file extensions (.fasta or .yaml)
        # If the input file has a non-standard extension, copy it with the correct one
        $local_input = ensure_correct_extension($local_input, $input_format, $input_dir);
    }

    # Find boltz binary: check PATH first, then P3_BOLTZ_PATH, then default
    my $boltz_bin = find_boltz_binary();
    print "Using boltz binary: $boltz_bin\n";

    # Set BOLTZ_CACHE to a writable location for model weights and CCD data
    # Boltz downloads model weights and CCD dictionary on first run
    if (!$ENV{BOLTZ_CACHE}) {
        my $boltz_cache_dir = "$work_dir/boltz_cache";
        make_path($boltz_cache_dir) unless -d $boltz_cache_dir;
        $ENV{BOLTZ_CACHE} = $boltz_cache_dir;
        print "Boltz cache directory: $boltz_cache_dir\n";
    }

    # Build boltz command
    my @cmd = ($boltz_bin, "predict", $local_input);

    # Cache directory for model weights and CCD data
    push @cmd, "--cache", $ENV{BOLTZ_CACHE};

    # Output directory
    push @cmd, "--out_dir", $output_dir;

    # MSA server option
    if ($params->{use_msa_server} // 1) {
        push @cmd, "--use_msa_server";
    }

    # Diffusion samples
    if (my $samples = $params->{diffusion_samples}) {
        push @cmd, "--diffusion_samples", $samples;
    }

    # Recycling steps
    if (my $steps = $params->{recycling_steps}) {
        push @cmd, "--recycling_steps", $steps;
    }

    # Sampling steps
    if (my $sampling = $params->{sampling_steps}) {
        push @cmd, "--sampling_steps", $sampling;
    }

    # Output format
    if (my $format = $params->{output_format}) {
        push @cmd, "--output_format", $format;
    }

    # Use potentials
    if ($params->{use_potentials}) {
        push @cmd, "--use_potentials";
    }

    # Write PAE matrix
    if ($params->{write_full_pae}) {
        push @cmd, "--write_full_pae";
    }

    # Accelerator (default GPU)
    my $accelerator = $params->{accelerator} // "gpu";
    push @cmd, "--accelerator", $accelerator;

    # Execute boltz
    print "Executing: " . join(" ", @cmd) . "\n";

    my $rc = system(@cmd);
    if ($rc != 0) {
        die "Boltz prediction failed with exit code: $rc\n";
    }

    print "Boltz prediction completed successfully\n";

    # Get output folder from app framework
    my $output_folder = $app->result_folder();
    die "Could not get result folder from app framework\n" unless $output_folder;

    # Clean up trailing slashes/dots
    $output_folder =~ s/\/+$//;
    $output_folder =~ s/\/\.$//;

    # Use output_file parameter as base name, create unique subfolder
    my $output_base = $params->{output_file} // "boltz_result";
    my $timestamp = POSIX::strftime("%Y%m%d_%H%M%S", localtime);
    my $task_id = $app->{task_id} // "unknown";
    my $run_folder = "${output_base}_${timestamp}_${task_id}";
    $output_folder = "$output_folder/$run_folder";

    print "Uploading results to workspace: $output_folder\n";
    upload_results($app, $output_dir, $output_folder);

    # Also upload the sequence ID mapping file if it exists
    if ($mapping_file && -f $mapping_file) {
        print "Uploading sequence ID mapping: $mapping_file\n";
        my @cmd = ("p3-cp", "--overwrite", $mapping_file, "ws:$output_folder/sequence_id_mapping.json");
        system(@cmd);  # Don't fail if this doesn't work
    }

    print "Boltz job completed\n";
    return 0;
}

=head2 detect_input_format

Detect whether input is YAML or FASTA format.

=cut

sub detect_input_format {
    my ($file, $hint) = @_;

    # If format is explicitly specified and not "auto"
    if ($hint && $hint ne "auto") {
        return $hint;
    }

    # Detect by extension
    if ($file =~ /\.ya?ml$/i) {
        return "yaml";
    } elsif ($file =~ /\.fa(sta)?$/i) {
        return "fasta";
    }

    # Try to detect by content
    my $content = read_file($file, { binmode => ':raw' });
    if ($content =~ /^(version:|sequences:)/m) {
        return "yaml";
    } elsif ($content =~ /^>/m) {
        return "fasta";
    }

    # Default to yaml
    return "yaml";
}

=head2 ensure_correct_extension

Ensure input file has correct extension for Boltz (.fasta or .yaml).
Boltz requires specific file extensions to determine input type.
If the file has a non-standard extension, copy it with the correct one.

=cut

sub ensure_correct_extension {
    my ($file, $format, $output_dir) = @_;

    # Determine expected extension
    my $expected_ext = ($format eq "fasta") ? ".fasta" : ".yaml";

    # Check if file already has correct extension
    if ($file =~ /\Q$expected_ext\E$/i) {
        return $file;
    }

    # Also accept common variants that Boltz recognizes
    # Boltz accepts: .fa, .fas, .fasta for FASTA; .yml, .yaml for YAML
    if ($format eq "fasta" && $file =~ /\.(fa|fas)$/i) {
        return $file;
    }
    if ($format eq "yaml" && $file =~ /\.yml$/i) {
        return $file;
    }

    # Need to copy with correct extension
    my $basename = basename($file);
    $basename =~ s/\.[^.]+$//;  # Remove existing extension
    my $new_file = "$output_dir/${basename}${expected_ext}";

    print "Copying input to correct extension: $file -> $new_file\n";
    copy($file, $new_file) or die "Failed to copy file: $!\n";

    return $new_file;
}

=head2 rewrite_fasta_for_boltz

Rewrite FASTA file to use Boltz-compatible identifiers.

Boltz expects headers in format: >CHAIN_ID|ENTITY_TYPE
For example: >A|protein, >B|protein, >L|smiles, >D|dna

This function:
1. Reads the input FASTA
2. Assigns chain IDs (A, B, C, ...) to each sequence
3. Detects entity type (protein, dna, rna, smiles)
4. Writes a new FASTA with Boltz-compatible headers
5. Creates a mapping file to track original IDs

Returns: ($new_fasta_path, $mapping_file_path)

=cut

sub rewrite_fasta_for_boltz {
    my ($input_file, $output_dir) = @_;

    my $content = read_file($input_file, { binmode => ':raw' });

    # Parse sequences - split on header lines
    my @entries;
    my @blocks = split(/(?=^>)/m, $content);

    for my $block (@blocks) {
        next unless $block =~ /\S/;  # Skip empty blocks
        next unless $block =~ /^>/;  # Must start with >

        # Split header from sequence
        my ($header_line, @seq_lines) = split(/\n/, $block);

        # Extract header (remove leading >)
        my $header = $header_line;
        $header =~ s/^>//;
        $header =~ s/\s+$//;  # Trim trailing whitespace

        # Join sequence lines and remove all whitespace
        my $seq = join('', @seq_lines);
        $seq =~ s/\s+//g;

        if (length($seq) > 0) {
            push @entries, {
                original_header => $header,
                sequence => $seq,
            };
        } else {
            warn "Warning: Empty sequence for header: $header\n";
        }
    }

    die "No sequences found in FASTA file\n" unless @entries;

    # Assign chain IDs and detect entity types
    my @chain_ids = ('A'..'Z', 'a'..'z');  # Up to 52 chains
    my @mapping;
    my @new_fasta_lines;

    for my $i (0 .. $#entries) {
        my $entry = $entries[$i];
        my $chain_id = $chain_ids[$i] // die "Too many sequences (max 52 supported)\n";

        # Detect entity type from sequence content
        my $entity_type = detect_entity_type($entry->{sequence}, $entry->{original_header});

        # Build new header: >CHAIN_ID|ENTITY_TYPE
        my $new_header = "$chain_id|$entity_type";

        push @new_fasta_lines, ">$new_header\n$entry->{sequence}\n";

        push @mapping, {
            chain_id => $chain_id,
            entity_type => $entity_type,
            original_id => $entry->{original_header},
        };

        print "  Chain $chain_id ($entity_type): $entry->{original_header}\n";
    }

    # Write new FASTA with correct extension
    my $new_fasta = "$output_dir/boltz_input.fasta";
    write_file($new_fasta, join('', @new_fasta_lines));

    # Write mapping file (JSON)
    my $mapping_file = "$output_dir/sequence_id_mapping.json";
    write_file($mapping_file, encode_json(\@mapping));

    print "Rewrote FASTA for Boltz: $new_fasta\n";
    print "ID mapping saved to: $mapping_file\n";

    return ($new_fasta, $mapping_file);
}

=head2 detect_entity_type

Detect the entity type (protein, dna, rna, smiles) from sequence content.

=cut

sub detect_entity_type {
    my ($sequence, $header) = @_;

    # Check header for explicit type hints
    if ($header =~ /\|\s*(protein|dna|rna|smiles)\s*(\||$)/i) {
        return lc($1);
    }

    # Check if it's a SMILES string (contains special characters)
    if ($sequence =~ /[=#@\[\]\(\)\+\-]/ && $sequence =~ /^[A-Za-z0-9=#@\[\]\(\)\+\-\.\\\/%]+$/) {
        return 'smiles';
    }

    # Normalize sequence for analysis
    my $upper_seq = uc($sequence);

    # Count nucleotide vs amino acid characters
    my $dna_chars = ($upper_seq =~ tr/ATCG//);
    my $rna_chars = ($upper_seq =~ tr/AUCG//);
    my $protein_chars = ($upper_seq =~ tr/ACDEFGHIKLMNPQRSTVWY//);

    my $seq_len = length($sequence);
    return 'protein' if $seq_len == 0;  # Default for empty

    # If >90% DNA nucleotides (ATCG only), it's DNA
    if ($dna_chars / $seq_len > 0.9 && $upper_seq !~ /U/) {
        return 'dna';
    }

    # If contains U and >90% RNA nucleotides, it's RNA
    if ($upper_seq =~ /U/ && $rna_chars / $seq_len > 0.9) {
        return 'rna';
    }

    # Default to protein
    return 'protein';
}

=head2 download_workspace_file

Download a file from the BV-BRC workspace.

=cut

sub download_workspace_file {
    my ($app, $ws_path, $local_dir) = @_;

    my $basename = basename($ws_path);
    my $local_path = "$local_dir/$basename";

    # Use workspace API to download
    if ($app && $app->can('workspace')) {
        try {
            # use_shock=1 required for files > 1KB (stored in Shock automatically)
            $app->workspace->download_file($ws_path, $local_path, 1);
        } catch {
            die "Failed to download $ws_path: $_\n";
        };
    } else {
        # Fallback for testing without workspace
        if (-f $ws_path) {
            copy($ws_path, $local_path) or die "Copy failed: $!\n";
        } else {
            die "File not found: $ws_path\n";
        }
    }

    return $local_path;
}

=head2 upload_results

Upload prediction results to the BV-BRC workspace.

=cut

sub upload_results {
    my ($app, $local_dir, $ws_path) = @_;

    # Find all output files
    my @files;
    find_files($local_dir, \@files);

    my @mapping = ('--map-suffix' => "txt=txt",
                   '--map-suffix' => "pdb=pdb",
                   '--map-suffix' => "cif=cif",
                   '--map-suffix' => "mmcif=mmcif",
                   '--map-suffix' => "json=json",
                   '--map-suffix' => "npz=unspecified",
                   '--map-suffix' => "fasta=protein_feature_fasta",
                   '--map-suffix' => "fa=protein_feature_fasta",
                   '--map-suffix' => "faa=protein_feature_fasta");

    my @cmd = ("p3-cp", "--overwrite", "-r", @mapping, $local_dir, "ws:$ws_path");
    print "@cmd\n";
    my $rc = system(@cmd);
    $rc == 0 or die "Error copying data to workspace\n";
}

=head2 find_files

Recursively find all files in a directory.

=cut

sub find_files {
    my ($dir, $files) = @_;

    opendir(my $dh, $dir) or return;
    while (my $entry = readdir($dh)) {
        next if $entry =~ /^\./;
        my $path = "$dir/$entry";
        if (-d $path) {
            find_files($path, $files);
        } else {
            push @$files, $path;
        }
    }
    closedir($dh);
}

=head2 find_boltz_binary

Find the boltz binary. Checks in order:
1. boltz in PATH
2. P3_BOLTZ_PATH environment variable
3. Default path /opt/conda-boltz/bin

=cut

sub find_boltz_binary {
    my $binary = "boltz";

    # Check if boltz is in PATH by iterating PATH entries
    if (my $path_env = $ENV{PATH}) {
        my @path_dirs = split(/:/, $path_env);
        for my $dir (@path_dirs) {
            next unless $dir;  # Skip empty entries
            my $full_path = "$dir/$binary";
            if (-x $full_path && !-d $full_path) {
                return $full_path;
            }
        }
    }

    # Check P3_BOLTZ_PATH environment variable
    if (my $boltz_path = $ENV{P3_BOLTZ_PATH}) {
        my $bin_path = "$boltz_path/$binary";
        if (-x $bin_path) {
            return $bin_path;
        }
    }

    # Default to /opt/conda-boltz/bin
    $ENV{P3_BOLTZ_PATH} //= "/opt/conda-boltz/bin";
    return "$ENV{P3_BOLTZ_PATH}/$binary";
}

__END__

=head1 AUTHOR

BV-BRC Team

=head1 LICENSE

MIT License

=cut
