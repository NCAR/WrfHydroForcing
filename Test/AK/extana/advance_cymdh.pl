#!/usr/bin/perl
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* 
# ** Copyright UCAR (c) [RAP] 1996 - 2003. All Rights Reserved. 
# *=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* 

## release notes - advance_cymdh.pl
# 4dwx release 1.0
# author - Becky Ruttenberg
# date - December 27, 2001

# function - this script duplicates the functionality of advance_cymdh.exe
# (written in FORTRAN) without leaving input and output files lying around.
# Given a date and amount to increment, it advances the date that amount, 
# taking end of day, month, year and leap years into account.

# cvs location - /cvs/projects/4dwx/archive/bin
# install location - /home/archive/bin on the DACs
#                    /home/fddasys/datbin on the master node of the RACs
# to call - advance_cymdh.pl date increment where date is the 4-digit year +
#   2-digit month + 2-digit day + 2-digit hour and increment is the amount in
#   hours to advance the date (give a negative number to go backwards in time).
# examples - advance_cymdh.pl 2001122700 24 returns 2001122800
#            advance_cymdh.pl 2001122700 -48 returns 2001122500

## other comments
# this script returns a number that is meant to be used from calling scripts.
# It therefore doesn't append a new-line character, making it non user-friendly
# from the command line.

## begin code
# required Perl module - strict is in the standard Perl distribution
use strict;            # enforces that variables are declared

my ($ccyymmddhh, $dh) = @ARGV;

my $ccyy = int ($ccyymmddhh / 1000000);
my $mmddhh = $ccyymmddhh % 1000000;
my $mm = int ($mmddhh / 10000);
my $dd = int (($mmddhh % 10000) / 100);
my $hh = $mmddhh % 100;
my @mmday = qw(31 28 31 30 31 30 31 31 30 31 30 31);

$hh += $dh;

#print "old hh is $hh, old dd is $dd\n";

while (($hh < 0) || ($hh > 23)) {
  if ($hh < 0) {
    $hh += 24;
    change_date(-1);
  } elsif ($hh > 23) {
    $hh -= 24;
    change_date(1);
  }
}

printf("%04d%02d%02d%02d", $ccyy, $mm, $dd, $hh);

sub change_date {
  my $delta = $_[0];

  if (($ccyy % 4) == 0) {
    if (($ccyy % 400) == 0) {
       $mmday[1] = 29;
    } elsif (($ccyy % 100) != 0) {
       $mmday[1] = 29;
    }
  }

  $dd += $delta;
  if ($dd == 0) {
    $mm -= 1;
    if ($mm == 0) {
      $mm = 12;
      $ccyy -= 1;
    }
    $dd = $mmday[$mm-1];
  } elsif ($dd > $mmday[$mm-1]) {
    $dd = 1;
    $mm += 1;
    if ($mm > 12) {
      $mm = 1;
      $ccyy += 1;
    }
  }
}
