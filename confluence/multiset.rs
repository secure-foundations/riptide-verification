use vstd::prelude::*;
use vstd::multiset::Multiset;
use crate::semantics::*;

verus! {

/**
 * Defines s1 >= s2 as a multiset
 */
pub open spec fn multiset_contains(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>) -> bool
{
    s2.to_multiset().subset_of(s1.to_multiset())
}

pub proof fn lemma_multiset_contains_length(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>)
    requires
        multiset_contains(s1, s2),

    ensures
        s1.len() >= s2.len(),
{
    s1.to_multiset_ensures();
    s2.to_multiset_ensures();
    let _ = s1.to_multiset().sub(s2.to_multiset()).len();
}

pub proof fn lemma_multiset_singleton_seq(op: OperatorIndex)
    ensures
        seq![op].to_multiset() =~= Multiset::singleton(op),
{
    assert(seq![op][0] == op);
    seq![op].to_multiset_ensures();
}

pub proof fn lemma_multiset_add_commutes_with_concat(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>)
    ensures
        (s1 + s2).to_multiset() == s1.to_multiset().add(s2.to_multiset()),
{
    vstd::seq_lib::lemma_multiset_commutative(s1, s2);
}

pub proof fn lemma_multiset_remove_insert(s: Seq<OperatorIndex>, i: int)
    requires
        0 <= i < s.len(),

    ensures
        s.remove(i).to_multiset().insert(s[i]) == s.to_multiset(),
{
    s.to_multiset_ensures();

    let _ = s.to_multiset().count(s[i]);

    let s1 = s.subrange(0, i);
    let s2 = seq![s[i]];
    let s3 = s.skip(i + 1);

    assert(s =~= s1 + s2 + s3);

    vstd::seq_lib::lemma_multiset_commutative(s1 + s2, s3);
    vstd::seq_lib::lemma_multiset_commutative(s1 + s3, s2);
    vstd::seq_lib::lemma_multiset_commutative(s1, s2);
    vstd::seq_lib::lemma_multiset_commutative(s1, s3);

    assert(s.remove(i).to_multiset() == s1.to_multiset().add(s3.to_multiset()));

    (s1 + s3).to_multiset_ensures();

    assert(s.remove(i).to_multiset().insert(s[i]) == s.to_multiset());
}

pub proof fn lemma_multiset_contains_remove(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>, i: int, j: int)
    requires
        0 <= i < s1.len(),
        0 <= j < s2.len(),
        s1[i] == s2[j],

    ensures
        multiset_contains(s1, s2) <==> multiset_contains(s1.remove(i), s2.remove(j)),
{
    lemma_multiset_remove_insert(s1, i);
    lemma_multiset_remove_insert(s2, j);
    assert(s1.remove(i).to_multiset() == s1.to_multiset().remove(s1[i]));
    assert(s2.remove(j).to_multiset() == s2.to_multiset().remove(s2[j]));
}

/**
 * If two sequences have the same prefixes, then removing them
 * would not affect the containment relation between them
 */
pub proof fn lemma_multiset_contains_remove_prefix(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>, n: int)
    requires
        0 <= n <= s1.len(),
        0 <= n <= s2.len(),
        forall |i: int| 0 <= i < n ==> s1[i] == s2[i],

    ensures
        multiset_contains(s1, s2) <==> multiset_contains(s1.skip(n), s2.skip(n)),

    decreases n,
{
    if n > 0 {
        lemma_multiset_contains_remove(s1, s2, 0, 0);
        lemma_multiset_contains_remove_prefix(s1.remove(0), s2.remove(0), n - 1);

        assert(s1.remove(0).skip(n - 1) =~= s1.skip(n));
        assert(s2.remove(0).skip(n - 1) =~= s2.skip(n));
    } else {
        assert(s1.skip(0) =~= s1);
        assert(s2.skip(0) =~= s2);
    }
}

pub proof fn lemma_multiset_not_contains_singleton(s1: Seq<OperatorIndex>, op: OperatorIndex)
    requires
        !multiset_contains(s1, seq![op]),

    ensures
        !s1.contains(op),
{
    s1.to_multiset_ensures();
    seq![op].to_multiset_ensures();
}

pub proof fn lemma_multiset_contains_transitive(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>, s3: Seq<OperatorIndex>)
    requires
        multiset_contains(s1, s2),
        multiset_contains(s2, s3),

    ensures
        multiset_contains(s1, s3),
{}

pub proof fn lemma_multiset_contains_transitive_elem(s1: Seq<OperatorIndex>, s2: Seq<OperatorIndex>, op: OperatorIndex)
    requires
        multiset_contains(s1, s2),
        s2.contains(op),

    ensures
        s1.contains(op),
{
    s1.to_multiset_ensures();
    s2.to_multiset_ensures();
    let _ = s1.to_multiset().count(op);
}

pub closed spec fn multiset_split(a: Seq<OperatorIndex>, b: Seq<OperatorIndex>) -> int
    decreases b.len()
{
    if b.len() == 0 || !a.to_multiset().contains(b[0]) {
        0
    } else {
        let i = a.index_of(b[0]);
        multiset_split(a.remove(i), b.remove(0)) + 1
    }
}

/**
 * Lemma: If a does not contain b, then there exists a split point within b such that
 * a contains b[:split] and does not contain b[split:]
 *
 * Such split is witnessed by multiset_split(a, b)
 */
pub proof fn lemma_multiset_split(a: Seq<OperatorIndex>, b: Seq<OperatorIndex>)
    requires !multiset_contains(a, b)
    ensures ({
        let split = multiset_split(a, b);
        0 <= split < b.len() &&
        split <= a.len() &&
        multiset_contains(a, b.take(split)) &&
        !multiset_contains(a, b.take(split + 1))
    })
    decreases b.len()
{
    if b.len() > 0 && a.to_multiset().contains(b[0]) {
        a.to_multiset_ensures();
        assert(a.contains(b[0]));

        let i = a.index_of(b[0]);
        lemma_multiset_contains_remove(a, b, i, 0);

        let split = multiset_split(a.remove(i), b.remove(0)) + 1;
        lemma_multiset_split(a.remove(i), b.remove(0));

        assert(b.remove(0).take(split - 1) =~= b.take(split).remove(0));
        assert(b.remove(0).take(split) =~= b.take(split + 1).remove(0));

        // assert(multiset_contains(a.remove(i), b.take(split).remove(0)));
        // assert(!multiset_contains(a.remove(i), b.take(split + 1).remove(0)));

        lemma_multiset_contains_remove(a, b.take(split), i, 0);
        lemma_multiset_contains_remove(a, b.take(split + 1), i, 0);
    } else {
        a.to_multiset_ensures();
        b.to_multiset_ensures();
        b.take(0).to_multiset_ensures();
        lemma_multiset_singleton_seq(b[0]);
        assert(seq![b[0]] =~= b.take(1));

        if b.len() == 0 {
            assert(multiset_contains(a, b));
            assert(false);
        }

        // assert(!a.to_multiset().contains(b[0]));
        // assert(!Multiset::singleton(b[0]).subset_of(a.to_multiset()));
        // assert(seq![b[0]].to_multiset() == Multiset::singleton(b[0]));
    }
}

}
