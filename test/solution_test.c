/* file minunit_example.c */

#include <stdio.h>
#include "minunit.h"
#include "../solution.h"

int tests_run = 0;

int foo = 7;
int bar = 4;

static char * testLongestPalindromeReturnsZeroForEmptyString() {
    mu_assert("error, longestPalindrome with an empty string as input doesn't equal 0", longestPalindrome("") == 0);
    return 0;
}

static char * testLongestPalindromeReturnsOneForSingleCharacter() {
    mu_assert("error, longestPalindrome with a single character doesn't equal 1", longestPalindrome("a") == 1);
    return 0;
}

static char * testLongestPalindromeReturnsZeroForSingleSpaceCharacter() {
    mu_assert("error, longestPalindrome with a single space character doesn't equal 0", longestPalindrome(" ") == 0);
    return 0;
}

static char * testLongestPalindromeReturnsThreeForStringaaba() {
    mu_assert("error, longestPalindrome for string 'aaba' doesn't equal 3", longestPalindrome("aaba") == 3);
    return 0;
}

static char * testLongestPalindromeReturnsFourForStringaaaa() {
    mu_assert("error, longestPalindrome for string 'aaaa' doesn't equal 4", longestPalindrome("aaaa") == 4);
    return 0;
}

static char * all_tests() {
    mu_run_test(testLongestPalindromeReturnsZeroForEmptyString);
    mu_run_test(testLongestPalindromeReturnsOneForSingleCharacter);
    mu_run_test(testLongestPalindromeReturnsZeroForSingleSpaceCharacter);
    mu_run_test(testLongestPalindromeReturnsThreeForStringaaba);
    mu_run_test(testLongestPalindromeReturnsFourForStringaaaa);
    return 0;
}

int main(int argc, char **argv) {
    char *result = all_tests();
    if (result != 0) {
        printf("%s\n", result);
    }
    else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);

    return result != 0;
}
